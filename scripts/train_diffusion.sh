ENT="python diffusion.py "
cfg="config/DiffusionVAE.yaml"
saveroot=""
exp_name=""
launcher="none"
vae_checkpoint=""



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --vae_checkpoint $vae_checkpoint \

