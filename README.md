# Neural-Style-Transfer

Neural Style Transfer is the state of the state-of-the-art technique to generate beutiful images by taking a Content Image and mixing it with what is called as a Style Image

let's look at an example:

we consider the following image to our content image ("Nivin Pauly is an Indian actor and producer. Known for his work in Malayalam cinema")

![](Images/name.jpeg)

And say the following is our style Image:

![](Images/style.jpeg)

Using the Neural Style Transfer as explained in the paper we can get a generated image as teh following Image:

![](Images/generated1.png)

As you can see it has taken the content from the content image and applied a style to it to generate a new image

IF you have a look in the code due to computing restrictions we reduced the image size of the generated image to 356, you can change the same if you have good computational resourses(just change the value of "img_size" variable in teh code

IF you want more information we highly recommend you to read the paper 
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)


Also following [video](https://www.youtube.com/watch?v=imX4kSKDY7s&t=1031s) also helped us alot 
