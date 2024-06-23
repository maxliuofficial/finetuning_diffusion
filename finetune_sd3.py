import math
import os
from dataclasses import dataclass

import accelerate.utils
import click
import datasets
import torch
import torch.nn.functional as F
import torch.utils.data
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

OUTPUT_BASEDIR = "output_dir"


class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        instance_prompt: str,
        size: int = 512,
    ):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = {}
        image = self.dataset[index]["images"]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt"] = self.instance_prompt
        return example


@dataclass
class TrainingArgs:
    instance_prompt: str
    learning_rate: float = 2e-06
    max_train_steps: int = 400
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 1  # Increase to save memory
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True  # True to save memory
    use_8bit_adam: bool = True  # Use 8bit optimizer from bitsandbytes
    seed: int = 1000101
    sample_batch_size: int = 2


def build_instance_prompt(name: str, typ: str) -> str:
    instance_prompt = f"a photo of {name} {typ}"
    print(f"Instance prompt: {instance_prompt}")
    return instance_prompt


def load_dataset_from_folder(folder_path) -> datasets.Dataset:
    # Lists to store the data
    images = []
    # Loop through each subfolder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        # Open the image file
        with Image.open(image_path) as img:
            # Append the image and its label to the lists
            images.append(img.convert("RGB"))

    return datasets.Dataset.from_dict({"images": images})


def collate(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    instance_prompts = [example["instance_prompt"] for example in examples]
    pixels_list = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixels_list)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"instance_prompts": instance_prompts, "pixel_values": pixel_values}
    return batch


def train_model(
    training_args: TrainingArgs,
    dataset: datasets.Dataset,
    diffusion_pipe: StableDiffusion3Pipeline,
    lora_config: LoraConfig | None = None,
) -> SD3Transformer2DModel:
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )
    print(f"Training on {accelerator.device}.")
    accelerate.utils.set_seed(training_args.seed)

    # Get the text embedding for conditioning
    print("Encoding prompt...")
    diffusion_pipe_cuda = diffusion_pipe.to(accelerator.device)
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = diffusion_pipe_cuda.encode_prompt(
            training_args.instance_prompt, None, None
        )
    torch.cuda.empty_cache()

    transformer: SD3Transformer2DModel = diffusion_pipe.transformer
    # Save some memory
    if training_args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    trainable_params = transformer.parameters()
    # Maybe use lora if needed
    if lora_config is not None:
        transformer.add_adapter(lora_config)
        trainable_params = filter(lambda p: p.requires_grad, transformer.parameters())

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if training_args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        trainable_params,
        lr=training_args.learning_rate,
    )

    noise_scheduler: FlowMatchEulerDiscreteScheduler = diffusion_pipe.scheduler

    train_dataloader = DataLoader(
        dataset=TorchDataset(dataset, training_args.instance_prompt),
        batch_size=training_args.train_batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    print("Preparing accelerator...")
    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader
    )
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(transformer, SD3Transformer2DModel)
    assert isinstance(optimizer, torch.optim.Optimizer)

    # Move vae to gpu
    vae: AutoencoderKL = diffusion_pipe.vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        training_args.max_train_steps / num_update_steps_per_epoch
    )

    progress_bar = tqdm(
        range(training_args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                with torch.no_grad():
                    latents: torch.Tensor = vae.encode(
                        batch["pixel_values"].to(dtype=vae.dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(
                    device=accelerator.device, dtype=latents.dtype
                )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                indices = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                ).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)

                sigmas = noise_scheduler.sigmas.to(
                    device=accelerator.device, dtype=latents.dtype
                )
                schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                step_indices = [
                    (schedule_timesteps == t).nonzero().item() for t in timesteps
                ]

                sigma = sigmas[step_indices].flatten()
                while len(sigma.shape) < latents.ndim:
                    sigma = sigma.unsqueeze(-1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = sigma * noise + (1.0 - sigma) * latents

                # Predict the noise residual
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, training_args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= training_args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    return accelerator.unwrap_model(transformer)


def save_pipeline(
    output_dir: str,
    diffusion_pipe: StableDiffusion3Pipeline,
    transformer: SD3Transformer2DModel,
    use_lora: bool,
) -> None:
    print(f"Saving to {output_dir}.")
    if use_lora:
        lora_wts = get_peft_model_state_dict(transformer)
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=output_dir, transformer_lora_layers=lora_wts
        )
    else:
        # Create the pipeline using the trained modules and save it
        diffusion_pipe.transformer = transformer
        diffusion_pipe.save_pretrained(output_dir)


@click.command()
@click.option("--dataset-name", required=True)
@click.option("--name", required=True)
@click.option("--typ", required=True)
@click.option("--output-dir", required=True)
@click.option("--use-lora", is_flag=True, default=False)
def main(dataset_name: str, name: str, typ: str, output_dir: str, use_lora: bool):
    dataset = load_dataset_from_folder(dataset_name)
    instance_prompt = build_instance_prompt(name, typ)

    training_args = TrainingArgs(instance_prompt=instance_prompt)

    diffusion_pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
        token=os.getenv("hf_token"),
    )

    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
            r=128,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            # lora_dropout=0.1,  # TODO: try this out?
        )

    torch.cuda.empty_cache()

    finetuned_transformer = train_model(
        training_args, dataset, diffusion_pipe, lora_config
    )

    output_dir = f"{OUTPUT_BASEDIR}/{output_dir}"
    save_pipeline(output_dir, diffusion_pipe, finetuned_transformer, use_lora)

    print(f"Finished!")


if __name__ == "__main__":
    main()
