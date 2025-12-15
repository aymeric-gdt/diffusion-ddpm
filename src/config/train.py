from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 256
    image_channels = 3
    train_batch_size = 4 # Réduire à 2-4 si vous êtes sur CPU
    eval_batch_size = 6
    num_epochs = 50 # 150 normalement, 50 pour tester
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 5
    save_model_epochs = 5
    dataset = 'C:/Users/aymer/Documents/local-dev/deep-learing-advanced/diffusion-ddpm/landscape_dataset' # 
    output_dir = f'models/{dataset.split("/")[-1]}'
    device = "cpu" # cuda si GPU disponible
    seed = 0
    resume = None


training_config = TrainingConfig()