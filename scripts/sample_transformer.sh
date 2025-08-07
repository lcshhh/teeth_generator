ENT="python second_stage.py "
cfg="config/second_stage_sample.yaml"
saveroot=""
exp_name=""
ckpts=""
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpt $ckpts \
    --sample \

