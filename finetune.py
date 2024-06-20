import functools
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
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

OUTPUT_BASEDIR = "output_dir"


class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        instance_prompt: str,
        tokenizer: CLIPTokenizer,
        size: int = 512,
    ):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
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
        image = self.dataset[index]["image"]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            text=self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
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


def get_modules_for_lora(diffusion_pipe: StableDiffusionPipeline) -> list[str]:
    return [
        name
        for name, module in diffusion_pipe.unet.named_modules()
        # Only these are supported for lora
        if isinstance(
            module,
            (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d, torch.nn.Embedding),
        )
    ]


def collate(
    tokenizer: CLIPTokenizer, examples: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixels_list = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixels_list)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids

    batch = {"input_ids": input_ids, "pixel_values": pixel_values}
    return batch


def train_model(
    training_args: TrainingArgs,
    dataset: datasets.Dataset,
    diffusion_pipe: StableDiffusionPipeline,
    lora_config: LoraConfig | None = None,
) -> UNet2DConditionModel:
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )
    print(f"Training on {accelerator.device}.")
    accelerate.utils.set_seed(training_args.seed)

    unet: UNet2DConditionModel = diffusion_pipe.unet

    if training_args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trainable_params = unet.parameters()
    if lora_config is not None:
        unet.add_adapter(lora_config)
        trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

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

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    tokenizer: CLIPTokenizer = diffusion_pipe.tokenizer
    train_dataloader = DataLoader(
        dataset=TorchDataset(dataset, training_args.instance_prompt, tokenizer),
        batch_size=training_args.train_batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate, tokenizer),
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(unet, UNet2DConditionModel)
    assert isinstance(optimizer, torch.optim.Optimizer)

    # Move text_encode and vae to gpu
    text_encoder: CLIPTextModel = diffusion_pipe.text_encoder
    text_encoder.to(accelerator.device)
    vae: AutoencoderKL = diffusion_pipe.vae
    vae.to(accelerator.device)

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
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents: torch.Tensor = vae.encode(
                        batch["pixel_values"]
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), training_args.max_grad_norm
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

    return accelerator.unwrap_model(unet)


def save_pipeline(
    output_dir: str,
    diffusion_pipe: StableDiffusionPipeline,
    unet: UNet2DConditionModel,
) -> None:
    # Create the pipeline using the trained modules and save it
    print(f"Saving to {output_dir}.")
    diffusion_pipe.unet = unet
    diffusion_pipe.save_pretrained(output_dir)


@click.command()
@click.option("--dataset", required=True)
@click.option("--name", required=True)
@click.option("--typ", required=True)
@click.option("--output-dir", required=True)
@click.option("--use-lora", is_flag=True, default=False)
def main(dataset: str, name: str, typ: str, output_dir: str, use_lora: bool):
    dataset = datasets.load_dataset(dataset, split="train")
    instance_prompt = build_instance_prompt(name, typ)

    training_args = TrainingArgs(instance_prompt=instance_prompt)

    diffusion_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", token=os.getenv("hf_token")
    )

    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=get_modules_for_lora(diffusion_pipe),
            # lora_dropout=0.1,  # TODO: try this out?
        )

    finetuned_unet = train_model(training_args, dataset, diffusion_pipe, lora_config)

    output_dir = f"{OUTPUT_BASEDIR}/{output_dir}"
    save_pipeline(output_dir, diffusion_pipe, finetuned_unet)

    print(f"Finished!")


if __name__ == "__main__":
    main()
