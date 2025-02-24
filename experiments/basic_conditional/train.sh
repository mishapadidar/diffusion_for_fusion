# train on pca data
# python train_model.py --conditions 'mean_iota' 'aspect_ratio' --train_batch_size 4096 --num_epochs 500 --learning_rate 5e-4 --num_timesteps 200 --input_emb_size 64 --time_emb_size 128 --cond_emb_size 128 --hidden_size 512  --hidden_layers 3 --time_emb_type "sinusoidal" --use_pca --pca_size 9 --save_images_step 10
python train_model.py --conditions 'mean_iota' 'aspect_ratio' 'nfp' 'helicity' --train_batch_size 4096 --num_epochs 150 --learning_rate 5e-4 --num_timesteps 200 --input_emb_size 64 --time_emb_size 128 --cond_emb_size 128 --hidden_size 1024  --hidden_layers 3 --time_emb_type "sinusoidal" --use_pca --pca_size 9 --save_images_step 10

# train on all dimensions
# python train_model.py --conditions 'mean_iota' 'aspect_ratio' 'nfp' 'helicity' --train_batch_size 4096 --num_epochs 250 --learning_rate 5e-4 --num_timesteps 400 --input_emb_size 64 --time_emb_size 128 --cond_emb_size 128 --hidden_size 4096  --hidden_layers 3 --time_emb_type "sinusoidal" --save_images_step 10

