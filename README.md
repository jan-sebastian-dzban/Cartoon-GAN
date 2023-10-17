**6 Expirements**

Each of the models detailed in Chapter 3, with the new approach LightAnimeGAN described in Chapter 4, were built and trained using PyTorch in the Python language. Regrettably, due to the high computational demand and time-consuming nature of training GANs, we were only able to train all of the discussed models on the Miyazaki Hayao dataset. However, as we anticipated that the AnimeGAN and LightAnimeGAN models would likely perform the best, these models were trained on both the Hayao and Shinakai datasets (Chapter 5) for a comparative evaluation of style transformation. During the implementation and training process, we followed the original authors’ recommendations but also took measures to ensure a fair comparison among the different architectures by adopting as many uniform parameters and training techniques as possible.

In this section, we delve into the experiments, evaluation, and comparison of the models we described above. We explore each model individually, providing details on the implementation and training phases, as well as highlighting the unique parameters and characteristics inherent to each model. The software employed during the training phase is thoroughly discussed. We present the results achieved by each architecture, demonstrating their proficiency in converting real images into cartoons. The effectiveness of these conversions is visualized through combined plots and measured using Visual Inspection Analysis, User Study, and Frec´ het Inception Distance metrics. Additionally, we conduct a comparison of the style translations between the selected architectures.

**Software Used**

For the implementation, we utilized the Python language in version 3.7.12 [5] and GCCcore in version 8.3.0 (Debian 8.3.0-6).

**Libraries**

The key library used for implementing all the models is **PyTorch** , version 1.9.1 [35], originally developed by Meta AI and later overseen by the PyTorch Foundation in 2022. [3]. The PyTorch library supports Tensor processing units (TPUs) computation and the construction of Deep Neural Networks (DNNs) built on auto differentiation in Python with strong graphics processing unit (GPU) acceleration.

In performing various tasks, we made use of specialized libraries. Among these, the **tqdm** library was used to show the progress of training, **Pillow** was instrumental for image loading and manipulation, **NumPy** was crucial for conducting numerical operations including reshaping image data arrays, and dealing with intricate

matrix operations, while **Matplotlib** took on the task of visualizing images and graphing training metrics.

**Google Cloud Platform**

The model’s training process was conducted thanks to the Google Cloud Platform and their 12-month free trial with a $300 credit to use with any Google Cloud services. Google Cloud Platform (GCP) is, along with Amazon Web Services and Microsoft Azure, among the three dominant players in the public cloud market. GCP is a platform consisting of cloud computing services used for developing, testing, and deploying solutions using Google infrastructure [4]. For the purpose of this research, a virtual machine equipped with one NVIDIA V100 GPU, 4 vCPU, and 15 GB RAM was chosen.![ref1]


**Implementation and Training**

In the course of our experiments, we made a concerted effort to implement the parameters specified by the original authors. Our aim was to provide a fair comparison across the various architectures by employing as many consistent parameters and training techniques as possible. However, it’s important to note that our primary objective was to attain optimal results in cartoon generation, rather than simply replicating the results documented by the authors.

While the authors’ original parameters and training techniques served as our starting point, we found that in some instances, these did not yield the most desirable results for our specific dataset. In such cases, we

took the liberty to fine-tune and adjust the parameters with the goal of achieving improved results.

Our approach was iterative and experimental. When the original parameters fell short in generating satis- factory outputs, we viewed this as an opportunity to explore other alternatives. This adaptive and flexible approach allowed us to extract the best performance from the given architecture for our specific dataset. All models were trained and tested using the same dataset, as detailed in Chapter 5. The dataset was first prepared by organizing the data into the necessary folders and subfolders hierarchy as required by the model’s data loaders. The Leaky ReLU (LReLU) was employed across all models, utilizing the parameters: *α* = 0 *.*2. To manage the extent of model training, we constrained the training phase to a maximum of 100 epochs. Given that the training of GAN model is a highly nonlinear optimization task and can easily become trapped in a sub-optimal local minimum with random initialization we address this issue, following the authors of the papers, we propose an initialization phase to enhance the model’s convergence performance. It is essential to note that the generator network goal is to reconstruct the input photo in a cartoon style while maintaining the semantic content. The adversarial learning framework begins with a generator that reconstructs the content of input images. To this end, during the initialization phase, the generator network is pre-trained using only the semantic content loss *Lcon* (*G,D* ). Consistent with the authors recommendations for the chosen models,

an initialization phase comprising 10 initialization epochs was included in our study. During this phase, solely the content loss is utilized, which assists in stabilizing the training process.

**CycleGAN**

All of the experiments and training phases were conducted utilizing the repository shared by the CycleGAN authors. The PyTorch implementation of the repository can be accessed via this link: CycleGAN Repo [54]. [20, 11] have found that the inclusion of an additional loss function that encourages the preser- vation of color composition between input and output is beneficial for generating painting like photos. This approach uses the technique introduced by Taigman et al. [47] called identity loss depicted in Equation (6.1).

It aims to keep the generator close to an identity mapping when real samples from the target domain are used as input.

*Lident* (*G,F* ) = **E***y*∼ *p*data (*y*)[  *G*(*y*) − *y*  1] + **E***x*∼ *pdata* (*x*)[  *F*(*x*) − *x*  1] (6.1)

Without this identity loss *Lident* with*  1 distance function, the generators *G* and *F* could unnecessarily alter the color tone of input images. For instance, when translating between real photos and paintings, the generator could depict a daytime painting as a sunset photo, which is still valid under the adversarial and cycle consistency losses. This observation led us to believe that the identity loss would also be beneficial for generating cartoons, leading us to include it in the total loss function with the weight *λ ident* . This modification yields the following total loss function:

L(*G,F,D X ,DY* ) = L*GAN* (*G,D Y* ) + L*GAN* (*F,D X* ) + *λ cyc* L*cyc* (*G,F* ) + *λ ident* L*ident* (*G,F* ) (6.2)

In carrying out each experiment, we stayed faithful to the original author’s guidelines. We adjusted the parameters *λ cyc* and *λ ident* to 10 and 0.5, respectively, in accordance with Equation (3.6). We didn’t use any initialization epochs, instead, we trained all networks from scratch, with a learning rate that decreased

over time. The initial weights were assigned based on a Gaussian distribution *N* (0*,*0*.*02). We made use of the Adam optimizer, with a batch size set to 1 because of cycle consistency, a learning rate at 0.0002, and

*β*1 = 0 *.*5. According to the authors, this learning rate remained unchanged for the first 100 epochs and was then gradually brought down to zero over the next 100 epochs. However, since our training was limited to 100 epochs, we started applying linear decay after the 50th epoch.![ref1]


**GANILLA**

All of the experiments and training phases were executed using the repository shared by the GANILLA authors. The PyTorch implementation of the repository can be accessed via this link: GANILLA Repo [20]. that GANILLA is heavily inspired by CycleGAN model and shares the same loss functions, parameters,

and learning techniques. The entire training and implementation process is the same as the one mentioned above for CycleGAN (Section 6.2).

**CartoonGAN**

All of the experiments and training phases were executed using the repository shared by the CartoonGAN authors. The PyTorch implementation of the repository can be accessed via this link: CartoonGAN [11].

In the CartoonGAN paper, Chen et al. proposed an initialization phase to stabilize the training process. This phase, conducted over ten epochs, utilized the Adam optimizer with a learning rate of 0.0001.

Our initial attempt at training the CartoonGAN model adhered to the hyperparameters suggested in the original paper. Parameter *ω* in Equation (3.9) was set to 10, which according to the CartoonGAN authors, achieves an optimal balance between style and content preservation. Each training iteration was conducted on a batch of 4 images, and the Adam optimizer was set to a constant learning rate of 0.0003 with a weight decay of 0.0001.

Despite strict adherence to these parameters, the results were unsatisfactory. The generated images were almost indistinguishable from the original images, contradicting our intent of creating a distinct style trans- formation. This issue is evident in Figure 6.1, where the left-hand images are the original input photos and the right-hand images are the cartoon renditions generated by our model using the original paper’s parameters. The lack of cartoon like characteristics in the transformed images clearly indicates that our model did not perform as expected. This inconsistency may have arisen due to our use of a different dataset than the original authors. Our dataset contained four times fewer animation images compared to the original paper’s dataset. Interestingly, we were not alone in encountering issues when attempting to replicate the original results using![ref1] the specified parameters [45]. Inspired by similar challenges faced by other researchers, we interpreted these issues as inherent characteristics of the network. Consequently, we strived to tune the parameters for optimal results.

This prompted us to experiment further and fine tune the hyperparameters to better suit our dataset. Sub- sequent changes resulted in significant improvements in the model’s performance in cartoonization. However, these adjustments led to a decrease in the model’s performance in content preservation. A key modifica- tion was replacing instance normalization with layer normalization in both the discriminator and generator. The decision was based on the significant performance difference and was validated by comparing the loss functions of the two normalization techniques.

Following a series of experiments, we refined the hyperparameters. Specifically, we adjusted the *ω* parameter to 1.5 and increased the learning rate of the Adam optimizer to 0.0002. These adjustments resulted in an enhanced style transfer, better emulating the cartoon aesthetics while still sufficiently preserving the content from the original images. However, it’s important to note that the content often appears somewhat blurred and smudgy. Throughout the training process, we maintained a consistent learning rate of 0.0003 and a weight decay of 0.0001, executing each training iteration on a batch of 4 images.

**AnimeGAN**

All of the experiments and training phases were executed using the repository shared by the AnimeGAN authors. The PyTorch implementation of the repository can be accessed via this link: AnimeGAN Repo [9].  AnimeGAN model was implemented by adhering to specific procedural steps. Similar to the approach

in CartoonGAN, an initialization process was conducted over ten epochs, with a learning rate set to 0.0001. Subsequent training of the AnimeGAN was characterized by distinct learning rates for the generator and the discriminator, set at 0.00008 and 0.00016 respectively. The training process spanned 100 epochs, utilizing a

batch size of 4. To optimize the minimization of total loss, we employed the Adam optimizer.

In all experiments, the balance between style and content preservation was carefully regulated through the adjustment of the weights in Equation (3.13). These were used the same as in orginal AnimeGAN implemen- tation set as *ω*adv = 300, *ω*con = 1 *.*5, *ω*gra = 3, and *ω*col = 10.

Lastly, the scaling factor *ωs* in Equation (3.18) was maintained at 0.1. This helped to prevent the edges of

the generated image from appearing excessively sharp.

**LightAnimeGAN**

All of the experiments and training phases were executed using the repository shared by the LightAnimeGAN authors. The PyTorch implementation of the repository can be accessed via this link: LightAnimeGAN Repo.Given that LightAnimeGAN is heavily inspired by AnimeGAN model and shares the same loss functions, parameters, and learning techniques. The entire training and implementation process is the same as the one mentioned above for AnimeGAN (Section 6.2).

3. **Overall Comparison**

As shown in Table 6.2, there are differences between the considered models in terms of the number of parameters in the generator network and the training time required for one epoch on the described dataset.



||No of parameters G (10|6)|Generator size (MB)|Mean train time per epoch (sec)|
| :- | - | - | - | - |
|CycleGAN|11\.4||43\.4|4408|
|GANILLA|7\.2||27\.5|2834|
|CartoonGAN|12\.1||46\.7|4498|
|AnimeGAN|39\.6||15\.8|3780|
|**LightAnimeGAN**|**2.2**||**8.6**|**2263**|

Table 6.1: Comparison of models characteristics trained on the Hayao dataset.![ref1]

From models taken from the literature, the GANILLA had the least number of parameters with only 7.2 million. This leads to a smaller size of the generator model at only 27.5MB, which can be advantageous in terms of the memory resources required to store and deploy the model. The training time for GANILLA was also relatively shorter than other models at 2834 seconds per epoch.

On the other hand, the AnimeGAN model had the highest number of parameters at 39.6 million, resulting in a larger model size of 15.8MB. However, despite the larger model size and the higher number of parameters, the generator size and the training time per epoch for the AnimeGAN were shorter than both CycleGAN and CartoonGAN.

CycleGAN and CartoonGAN had a moderate number of parameters and generator size, but they required the longest training time per epoch at 4408 seconds and 4498 seconds respectively. The long training time could potentially slow down the model development and tuning process.

The LightAnimeGAN model, as suggested by its name, features the smallest generator size of just 8.6MB, and had the shortest mean training time per epoch at 2263 seconds. However, it’s noteworthy that the substantial difference in the number of parameters compared to other models does not correspondingly translate into large variations across other categories.

It should be noted that while the number of parameters, model size, and training time are important factors to consider when choosing a model, they do not dictate the performance of the model in generating cartoon- like images. Other factors such as the quality of the output images and the ability of the model to capture the unique characteristics of the target domain are the most crucial factor.

