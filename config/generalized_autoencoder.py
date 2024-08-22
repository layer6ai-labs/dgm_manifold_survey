def get_base_config(dataset, standalone):
    if standalone:
        standalone_info = {
            "train_batch_size": 128,
            "valid_batch_size": 128,
            "test_batch_size": 128,

            "make_valid_loader": True,

            "data_root": "data/",
            "logdir_root": "runs/"
        }
    else:
        standalone_info = {}

    return {
        "flatten": True,
        "denoising_sigma": None,
        "dequantize": False,
        "scale_data": True,
        "scale_data_to_n1_1": False,
        "whitening_transform": False,

        "optimizer": "adam",
        "lr": 0.001,
        "use_lr_scheduler": False,
        "max_epochs": 100,
        "max_grad_norm": 10,

        "early_stopping_metric": None,
        "max_bad_valid_epochs": None,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error"],

        **standalone_info
    }



def get_ae_config(dataset, standalone):
    net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"

    ae_base = {
        "latent_dim": 20,
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False
        }

    return {
        **ae_base,
        **net_config,
    }


def get_avb_config(dataset, standalone):
    net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"

    avb_base = {
        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 10,

        "latent_dim": 20,

        "discriminator_hidden_dims": [256, 256],

        "fixed_sigma": False,
        "single_sigma": True,

        "input_sigma": 3.,
        "prior_sigma": 1.,

        "lr": None,
        "disc_lr": 0.001,
        "nll_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_nll_lr_scheduler": True,
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True,
            "max_epochs": 50,
            "noise_dim": 256,
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False,
            "max_epochs": 100,
            "noise_dim": 256,
        }

    return {
        **avb_base,
        **net_config,
    }


def get_bigan_config(dataset, standalone):
    net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"

    bigan_base = {
        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 50,

        "latent_dim": 20,

        "discriminator_hidden_dims": [256, 256],
        "num_discriminator_steps": 2,
        "wasserstein": True,
        "clamp": 0.01,
        "gradient_penalty": True,
        "lambda": 10.0,
        "recon_weight": 1.0,

        "optimizer": 'adam',
        "lr": None,
        "disc_lr": 0.0001,
        "ge_lr": 0.0001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_ge_lr_scheduler": True,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error", "fid"],
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True,
            "max_epochs": 200,
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False,
            "max_epochs": 200,
        }

    return {
        **bigan_base,
        **net_config,
    }


def get_vae_config(dataset, standalone):
    # net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"
    net = "resnet"

    if net == "mlp":
        net_config = {
            "latent_dim": 20,
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "single_sigma": True,

            "flatten": True
        }

    elif net == "cnn":
        net_config = {
            "latent_dim": 20,
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "single_sigma": True,

            "flatten": False
        }

    elif net == "resnet":
        output_channels = 4
        ch_mults = (1, 2, 2)

        image_size = 28 if dataset in ["mnist", "fashion-mnist"] else 32
        latent_size = image_size // 2**(len(ch_mults) - 1)
        net_config = {
            "latent_dim": latent_size**2 * output_channels,
            "latent_shape": (output_channels, latent_size, latent_size),
            "encoder_net": "resnet",
            "decoder_net": "resnet",
            "output_channels": output_channels,
            "n_channels": 64,
            "ch_mults": ch_mults,
            "n_blocks": 2,

            "fixed_sigma": 0.05,

            "scale_data": False,
            "scale_data_to_n1_1": True,
            "dequantize": True,
            "flatten": False,

            "lr": 0.001,
            "max_epochs": 2000,

            "early_stopping_metric": "l2_reconstruction_error",
            "max_bad_valid_epochs": 50,
        }

    return {
        **net_config
    }


def get_adv_vae_config(dataset, standalone):
    net = "resnet"

    output_channels = 4
    ch_mults = (1, 2, 2)

    image_size = 28 if dataset in ["mnist", "fashion-mnist"] else 32
    latent_size = image_size // 2**(len(ch_mults) - 1)
    net_config = {
        "latent_dim": latent_size**2 * output_channels,
        "latent_shape": (output_channels, latent_size, latent_size),
        "encoder_net": "resnet",
        "decoder_net": "resnet",
        "discriminator_net": "resnet",
        "output_channels": output_channels,
        "n_channels": 64,
        "ch_mults": ch_mults,
        "n_blocks": 2,

        "beta1": 1e-6,
        "beta2": 1.,

        "scale_data": False,
        "scale_data_to_n1_1": True,
        "dequantize": True,
        "flatten": False,

        # "lr": 0.001,
        "optimizer": 'adam',
        "lr": None,
        "disc_lr": 0.0001,
        "ge_lr": 0.0001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_ge_lr_scheduler": True,
        "max_epochs": 2000,

        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 50,
        }

    return {
        **net_config
    }


def get_wae_config(dataset, standalone):
    net = "cnn"

    wae_base = {
        "latent_dim": 20,

        "max_epochs": 300,

        "discriminator_hidden_dims": [256, 256],

        "_lambda": 10.,
        "sigma": 1.,

        "lr": None,
        "disc_lr": 0.00025,
        "rec_lr": 0.0005,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": False,
        "use_rec_lr_scheduler": False,

        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 30,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error", "fid"],
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True
        }

    elif net == "cnn":
        enc_hidden_channels = [64, 64, 32, 32]
        enc_kernel = [3, 3, 3, 3]
        enc_stride = [1, 1, 1, 1]

        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": enc_hidden_channels,
            "encoder_kernel_size": enc_kernel,
            "encoder_stride": enc_stride,

            "decoder_net": "cnn",
            "decoder_hidden_channels": enc_hidden_channels[::-1],
            "decoder_kernel_size": enc_kernel[::-1],
            "decoder_stride": enc_stride[::-1],

            "flatten": False
        }

    return {
        **wae_base,
        **net_config,
    }


GAE_CFG_MAP = {
    "base": get_base_config,
    "ae": get_ae_config,
    "avb": get_avb_config,
    "bigan": get_bigan_config,
    "vae": get_vae_config,
    "adv_vae": get_adv_vae_config,
    "wae": get_wae_config
}
