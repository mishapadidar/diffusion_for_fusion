# train on pca data
# python ddpm_fusion.py --learning_rate 5e-4 --num_epochs 10000 --time_embedding "sinusoidal" --hidden_size 256  --hidden_layers 3 --embedding_size 64 --return_pca --train_batch_size 640 --save_images_step 100 --num_timesteps 200
# train on full size data
python ddpm_fusion.py --learning_rate 5e-4 --num_epochs 10000 --time_embedding "sinusoidal" --hidden_size 1024  --hidden_layers 3 --embedding_size 64 --train_batch_size 640 --save_images_step 100 --num_timesteps 600