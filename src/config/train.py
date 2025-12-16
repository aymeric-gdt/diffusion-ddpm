from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 256
    image_channels = 3
    train_batch_size = 6 # Réduire à 2-4 si vous êtes sur CPU
    eval_batch_size = 6
    num_epochs = 100 # 150 normalement, 50 pour tester
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 5
    save_model_epochs = 5
    dataset = 'landscape_dataset'
    output_dir = f'models/{dataset.split("/")[-1]}'
    device = "cuda" # cuda si GPU disponible
    seed = 0
    resume = 'models/landscape_dataset/unet256_e29.pth'


training_config = TrainingConfig()