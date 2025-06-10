# fig 11 data
# train on pca data
# python train_model.py --learning_rate 5e-4 --num_epochs 10000 --time_embedding "sinusoidal" --hidden_size 256  --hidden_layers 3 --embedding_size 64 --return_pca --train_batch_size 640 --save_images_step 100 --num_timesteps 200
# train on all dimensions
# python train_model.py --learning_rate 5e-4 --num_epochs 10000 --time_embedding "sinusoidal" --hidden_size 1024  --hidden_layers 5 --embedding_size 64 --train_batch_size 640 --save_images_step 100 --num_timesteps 300

# fig 9 data
# train on pca data
python train_model.py --learning_rate 5e-4 --num_epochs 10000 --time_embedding "sinusoidal" --hidden_size 256  --hidden_layers 3 --embedding_size 64 --return_pca --train_batch_size 640 --save_images_step 100 --num_timesteps 200 --dataset "fig9"
# train on all dimensions
# python train_model.py --learning_rate 5e-4 --num_epochs 10000 --time_embedding "sinusoidal" --hidden_size 1024  --hidden_layers 5 --embedding_size 64 --train_batch_size 640 --save_images_step 100 --num_timesteps 300 --dataset "fig9"
