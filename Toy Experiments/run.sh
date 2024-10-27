python main_toy.py \
    --targets_name "Ring" \
    --f_latent_dim 256 \
    --f_lr 0.005 \
    --p_item 2 \
    --num_particle 1000 \
    --f_iter 3 \
    --master_stepsize 0.01 \
    --n_epoch 2000 \
    --edge_width 0.1 \
    --requires_znet \
    --requiers_bound_loss
    
python main_toy.py \
    --targets_name "Cardioid" \
    --f_latent_dim 128 \
    --f_lr 0.002 \
    --p_item 2 \
    --num_particle 1000 \
    --f_iter 10 \
    --master_stepsize 0.005 \
    --n_epoch 2000 \
    --edge_width 0.1 \
    --requires_znet \
    --requiers_bound_loss

python main_toy.py \
    --targets_name "DoubleMoon" \
    --f_latent_dim 128 \
    --f_lr 0.002 \
    --p_item 2 \
    --num_particle 1000 \
    --f_iter 10 \
    --master_stepsize 0.005 \
    --n_epoch 2000 \
    --edge_width 0.1 \
    --requires_znet \
    --requiers_bound_loss