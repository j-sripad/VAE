![result](https://user-images.githubusercontent.com/39008190/119038179-727c0a80-b9d0-11eb-854b-8ac92c50d27c.jpeg)
#### Simple VAE implementation in pytorch 

##### Usage
> To run FashionMNIST
> > here Dataset can be FashionMNIST or MNIST
<pre><code>
python3 main.py --input_shape 28 28 --latent_dimension 20 --Dataset FashionMNIST --epochs 50 --batch_size 128
</code></pre>

> To generate synthetic Images from the trained model
<pre><code>
python3 generate_synthetic_data.py --input_shape 28 28 --latent_dimension 20 --batch_size 30 --model_path model/FashionMnist_model.pth --save_path generated_images/
</code></pre>
##### Sample results on FashionMNIST trained for 50 Epochs
