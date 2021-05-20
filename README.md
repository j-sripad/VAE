#### Simple VAE implementation in pytorch 

##### Usage
> To run FashionMNIST
<pre><code>
python3 main.py --input_shape 28 28 --latent_dimension 20 --Dataset FashionMNIST --epochs 50 --batch_size 128
>> Here Dataset can be FashionMNIST or MNIST
</code></pre>
> To generate synthetic Images from the trained model
<pre><code>
python3 generate_synthetic_data.py --input_shape 28 28 --latent_dimension 20 --batch_size 30 --model_path model/FashionMnist_model.pth --save_path generated_images/
</code></pre>
##### Sample results on FashionMNIST trained for 50 Epochs
