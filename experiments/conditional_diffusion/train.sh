# train on pca data with more data and bigger model
python train_model.py --conditions 'mean_iota' 'aspect_ratio' 'nfp' 'helicity' --train_batch_size 4096 --num_epochs 250 --learning_rate 5e-4 --num_timesteps 200 --input_emb_size 64 --time_emb_size 128 --cond_emb_size 128 --hidden_size 2048  --hidden_layers 4 --time_emb_type "sinusoidal" --use_pca --pca_size 200 --save_images_step 10
