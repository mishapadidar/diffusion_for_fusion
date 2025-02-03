# train on pca data
python train_model.py --conditions 'mean_iota' 'aspect_ratio' --train_batch_size 640 --num_epochs 10000 --learning_rate 5e-4 --num_timesteps 200 --input_emb_size 64 --time_emb_size 64 --cond_emb_size 64 --hidden_size 256  --hidden_layers 3 --time_emb_type "sinusoidal" --use_pca --save_images_step 100
# train on all dimensions
# python train_model.py --conditions 'mean_iota' 'aspect_ratio' --train_batch_size 640 --num_epochs 10000 --learning_rate 5e-4 --num_timesteps 200 --input_emb_size 64 --time_emb_size 64 --cond_emb_size 64 --hidden_size 1024  --hidden_layers 5 --time_emb_type "sinusoidal" --save_images_step 100
