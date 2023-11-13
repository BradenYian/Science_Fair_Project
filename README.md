# Science_Fair_Project

Adversarial learning is a machine learning paradigm where the model is trained to be robust against adversarial examples. Adversarial examples are inputs specifically crafted to deceive the model and cause misclassification. Defending against adversarial attacks is crucial for the deployment of machine learning models in real-world scenarios. Here's a literature review on defense methods for adversarial learning:

# Adversarial Training:
Description: Adversarial training involves augmenting the training dataset with adversarial examples, forcing the model to learn from both clean and adversarial data.
Key Studies:
Goodfellow, I., Shlens, J., & Szegedy, C. (2014). "Explaining and harnessing adversarial examples."
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards deep learning models resistant to adversarial attacks."


# Defensive Distillation:
Description: Defensive distillation involves training a model to mimic the behavior of a pre-trained model on clean data and then fine-tuning it with adversarial examples.
Key Studies:
Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2016). "Practical black-box attacks against machine learning."
Carlini, N., & Wagner, D. (2016). "Defensive distillation is not robust to adversarial examples."
Feature Squeezing:
Description: Feature squeezing reduces the precision of input features to make it harder for an attacker to craft effective adversarial examples.
Key Studies:
Xu, W., Evans, D., & Qi, Y. (2017). "Feature squeezing: Detecting adversarial examples in deep neural networks."
Randomization and Ensemble Methods:
Description: Introducing randomness during training or using ensemble methods can improve robustness against adversarial attacks by making it harder for attackers to generate effective adversarial examples.
Key Studies:
Liu, Y., Chen, X., Liu, C., & Song, D. (2018). "Delving into transferable adversarial examples and black-box attacks."
Liao, F., Liang, M., Dong, Y., Pang, T., & Hu, X. (2018). "Defense against adversarial attacks using feature scattering-based adversarial training."
# Certified Defense:
Description: Certified defense methods provide guarantees on the model's robustness by certifying that all inputs within a certain distance from the training data are classified correctly.
Key Studies:
Wong, E., Schmidt, L., Metzen, J. H., & Kolter, J. Z. (2018). "Scaling provable adversarial defenses."
Robust Optimization:
Description: Robust optimization involves modifying the training objective to explicitly account for adversarial perturbations.
Key Studies:
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards deep learning models resistant to adversarial attacks."
It's important to note that the field of adversarial learning is rapidly evolving, and new defense methods continue to be proposed. The effectiveness of these methods often depends on the specific application and the nature of the adversarial attacks the model may face.


# What is Adversarial Machine Learning Adversarial machine learning is a machine learning method that aims to trick machine learning models by providing deceptive input. Hence, it includes both the generation and detection of adversarial examples, which are inputs specially created to deceive classifiers. Such attacks, called adversarial machine learning, have been extensively explored in some areas, such as image classification and spam detection. 

# Adversarial Machine Learning example with FGSM, read more about this adversarial attack method below. – Source The most extensive studies of adversarial machine learning have been conducted in the area of image recognition, where modifications are performed on images that cause a classifier to produce incorrect predictions.   What is an Adversarial Example? An adversarial attack is a method to generate adversarial examples. Hence, an adversarial example is an input to a machine learning model that is purposely designed to cause a model to make a mistake in its predictions despite resembling a valid input to a human.   Difference between adversarial whitebox vs. blackbox attacks A whitebox attack is a scenario where the attacker has complete access to the target model, including the model’s architecture and its parameters. A blackbox attack is a scenario where an attacker has no access to the model and can only observe the outputs of the targeted model.   The Threat of Adversarial Attacks in Machine Learning With machine learning rapidly becoming core to organizations’ value proposition, the need for organizations to protect them is growing fast. Hence, Adversarial Machine Learning is becoming an important field in the software industry. Google, Microsoft, and IBM have started to invest in securing machine learning systems. In recent years, companies are heavily investing in machine learning themselves – Google, Amazon, Microsoft, and Tesla – faced some degree of adversarial attacks. The car with a camouflage pattern is misdetected as a “cake” – Source Moreover, governments start to implement security standards for machine learning systems, with the European Union even releasing a complete checklist to assess the trustworthiness of machine learning systems (Assessment List for Trustworthy Artificial Intelligence – ALTAI). Gartner, a leading industry market research firm, advised that “application leaders must anticipate and prepare to mitigate potential risks of data corruption, model theft, and adversarial samples”. Recent studies show that the security of today’s AI systems is of high importance to businesses. However, the emphasis is still on traditional security. Organizations seem to lack the tactical knowledge to secure machine learning systems in production. The adoption of a production-grade AI system drives the need for Privacy-Preserving Machine Learning (PPML).   How Adversarial Attacks on AI Systems Work There are a large variety of different adversarial attacks that can be used against machine learning systems. Many of these work on deep learning systems and traditional machine learning models such as Support Vector Machines (SVMs) and linear regression. Most adversarial attacks usually aim to deteriorate the performance of classifiers on specific tasks, essentially to “fool” the machine learning algorithm. Adversarial machine learning is the field that studies a class of attacks that aims to deteriorate the performance of classifiers on specific tasks. Adversarial attacks can be mainly classified into the following categories: Poisoning Attacks Evasion Attacks Model Extraction Attacks  

# Poisoning Attacks The attacker influences the training data or its labels to cause the model to underperform during deployment. Hence, Poisoning is essentially adversarial contamination of training data. As ML systems can be re-trained using data collected during operation, an attacker may poison the data by injecting malicious samples during operation, which subsequently disrupt or influence re-training. Adversarial Machine Learning Poisoning Attack – Source Evasion Attacks Evasion attacks are the most prevalent and most researched types of attacks. The attacker manipulates the data during deployment to deceive previously trained classifiers. Since they are performed during the deployment phase, they are the most practical types of attacks and the most used attacks on intrusion and malware scenarios. The attackers often attempt to evade detection by obfuscating the content of malware or spam emails. Therefore, samples are modified to evade detection as they are classified as legitimate without directly impacting the training data. Examples of evasion are spoofing attacks against biometric verification systems.   Model Extraction Model stealing or model extraction involves an attacker probing a black box machine learning system in order to either reconstruct the model or extract the data it was trained on. 

# This is especially significant when either the training data or the model itself is sensitive and confidential. Model extraction attacks can be used, for instance, to steal a stock market prediction model, which the adversary could use for their own financial benefit.   What Are Adversarial Examples? Adversarial examples are inputs to machine learning models that an attacker has purposely designed to cause the model to make a mistake. An adversarial example is a corrupted version of a valid input, where the corruption is done by adding a perturbation of a small magnitude to it. This barely noticed nuisance is designed to deceive the classifier by maximizing the probability of an incorrect class. The adversarial example is designed to appear “normal” to humans but causes misclassification by the targeted machine learning model. Following, we list some of the known current techniques for generating adversarial examples.   

# Popular Adversarial Attack Methods Limited-memory BFGS (L-BFGS) The Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method is a non-linear gradient-based numerical optimization algorithm to minimize the number of perturbations added to images. Advantages: Effective at generating adversarial examples. 
Disadvantages: Very computationally intensive, as it is an optimized method with box constraints. The method is time-consuming and impractical.  

# FastGradient Sign method (FGSM) A simple and fast gradient-based method is used to generate adversarial examples to minimize the maximum amount of perturbation added to any pixel of the image to cause misclassification. 
Advantages: Comparably efficient computing times. 
Disadvantages: Perturbations are added to every feature.   

# Jacobian-based Saliency Map Attack (JSMA) Unlike FGSM, the method uses feature selection to minimize the number of features modified while causing misclassification. Flat perturbations are added to features iteratively according to saliency value by decreasing order. 
Advantages: Very few features are perturbed.
 Disadvantages: More computationally intensive than FGSM.   

# Deepfool Attack This untargeted adversarial sample generation technique aims at minimizing the euclidean distance between perturbed samples and original samples. Decision boundaries between classes are estimated, and perturbations are added iteratively. Advantages: Effective at producing adversarial examples, with fewer perturbations and higher misclassification rates. 
Disadvantages: More computationally intensive than FGSM and JSMA. Also, adversarial examples are likely not optimal.   

# Carlini & Wagner Attack (C&W) The technique is based on the L-BFGS attack (optimization problem) but without box constraints and different objective functions. This makes the method more efficient at generating adversarial examples; it was shown to be able to defeat state-of-the-art defenses, such as defensive distillation and adversarial training. 
Advantages: Very effective at producing adversarial examples. Also, it can defeat some adversarial defenses. 
Disadvantages: More computationally intensive than FGSM, JSMA, and Deepfool.   Generative Adversarial Networks (GAN) 

# Generative Adversarial Networks (GANs) have been used to generate adversarial attacks, where two neural networks compete with each other. Thereby one is acting as a generator, and the other behaves as the discriminator. The two networks play a zero-sum game, where the generator tries to produce samples that the discriminator will misclassify. Meanwhile, the discriminator tries to distinguish real samples from ones created by the generator. Advantages: Generation of samples different from the ones used in training. 
Disadvantages: Training a Generate Adversarial Network is very computationally intensive and can be highly unstable.   

# Zeroth-order optimization attack (ZOO) The ZOO technique allows the estimation of the gradient of the classifiers without access to the classifier, making it ideal for black-box attacks. The method estimates gradient and hessian by querying the target model with modified individual features and uses Adam or Newton’s method to optimize perturbations. Advantages: Similar performance to the C&W attack. No training of substitute models or information on the classifier is required. 
Disadvantages: Requires a large number of queries to the target classifier.  
Read more at: https://viso.ai/deep-learning/adversarial-machine-learning/#
