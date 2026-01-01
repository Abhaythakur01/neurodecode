# Literature Review & References

## Essential Papers for Implementation

### Neural Decoding Fundamentals

1. **Kalman Filter for Neural Decoding**
   - Gilja et al. (2012). "A high-performance neural prosthesis enabled by control algorithm design." *Nature Neuroscience*.
   - Wu et al. (2006). "Neural Decoding of Cursor Motion Using a Kalman Filter." *NIPS*.

2. **Wiener Filter**
   - Saleh et al. (2010). "Fast Orthogonal Search for Training RBF Networks." *IEEE Transactions*.
   - Wessberg et al. (2000). "Real-time prediction of hand trajectory." *Nature*.

3. **Linear Discriminant Analysis**
   - Leuthardt et al. (2004). "A brain-computer interface using ECoG signals." *Journal of Neural Engineering*.

### Machine Learning Approaches

4. **Support Vector Machines**
   - Lal et al. (2004). "Support vector channel selection in BCI." *IEEE Transactions on Biomedical Engineering*.

5. **Random Forests & Ensemble Methods**
   - Flamary & Rakotomamonjy (2012). "Decoding finger movements from ECoG signals using Random Forests."

6. **XGBoost/Gradient Boosting**
   - Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.

7. **Gaussian Processes**
   - Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
   - Wu et al. (2009). "Bayesian Population Decoding of Motor Cortical Activity."

### Deep Learning for Neural Decoding

8. **RNN/LSTM Networks**
   - Sussillo et al. (2016). "LFADS - Latent Factor Analysis via Dynamical Systems." *Nature Communications*.
   - Glaser et al. (2020). "Machine Learning for Neural Decoding." *eNeuro*.

9. **Transformers for Neural Data**
   - Ye & Pandarinath (2021). "Representation learning for neural population activity with Neural Data Transformers." *NeurIPS*.
   - Azabou et al. (2023). "A Unified, Scalable Framework for Neural Population Decoding." *NeurIPS*.

10. **Temporal Convolutional Networks**
    - Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling."
    - Hammer et al. (2023). "Interpretable TCN for Multivariate Time Series."

11. **Variational Autoencoders**
    - Keshtkaran et al. (2021). "A large-scale neural network training framework for generalized estimation of single-trial population dynamics." *BioRxiv*.

### Meta-Learning & Adaptive Systems

12. **Model-Agnostic Meta-Learning (MAML)**
    - Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML*.

13. **Online Learning & Adaptation**
    - Orsborn et al. (2014). "Closed-loop decoder adaptation shapes neural plasticity for skillful neuroprosthetic control." *Neuron*.
    - Dangi et al. (2013). "Design and Analysis of Closed-Loop Decoder Adaptation Algorithms for Brain-Machine Interfaces."

14. **Ensemble Methods**
    - Dietterich (2000). "Ensemble Methods in Machine Learning." *International Workshop on Multiple Classifier Systems*.
    - Wolpaw & Wolpaw (2012). *Brain-Computer Interfaces: Principles and Practice*.

### Uncertainty Quantification

15. **Probabilistic Forecasting**
    - Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation." *ICML*.
    - Lakshminarayanan et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

### Clinical BCI Systems

16. **Clinical Trials**
    - Hochberg et al. (2012). "Reach and grasp by people with tetraplegia using a neurally controlled robotic arm." *Nature*.
    - Bouton et al. (2016). "Restoring cortical control of functional movement in a human with quadriplegia." *Nature*.

17. **Clinical Translation**
    - Collinger et al. (2013). "High-performance neuroprosthetic control by an individual with tetraplegia." *The Lancet*.
    - Benabid et al. (2019). "An exoskeleton controlled by an epidural wireless brain-machine interface in a tetraplegic patient." *The Lancet Neurology*.

### Datasets & Benchmarks

18. **Neural Latents Benchmark**
    - Pei et al. (2021). "Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity." *NeurIPS*.

19. **Public Datasets**
    - CRCNS: Collaborative Research in Computational Neuroscience data sharing
    - DANDI: Distributed Archives for Neurophysiology Data Integration
    - BCI Competition datasets

### Signal Processing & Preprocessing

20. **Spike Sorting**
    - Rey et al. (2015). "Past, present and future of spike sorting techniques." *Brain Research Bulletin*.
    - Pachitariu et al. (2016). "Kilosort: realtime spike-sorting for extracellular electrophysiology."

21. **Artifact Removal**
    - Mitra & Pesaran (1999). "Analysis of dynamic brain imaging data." *Biophysical Journal*.

### Real-Time Systems

22. **Real-Time Neural Decoding**
    - Gilja et al. (2015). "Clinical translation of a high-performance neural prosthesis." *Nature Medicine*.
    - Willett et al. (2021). "High-performance brain-to-text communication via handwriting." *Nature*.

### Regulatory & Ethics

23. **FDA Guidance**
    - FDA (2017). "Software as a Medical Device (SaMD): Clinical Evaluation."
    - ISO 14971:2019. "Medical devices — Application of risk management."

24. **Ethics & Privacy**
    - Ienca & Andorno (2017). "Towards new human rights in the age of neuroscience and neurotechnology." *Life Sciences, Society and Policy*.
    - Kellmeyer (2018). "Big Brain Data: On the Responsible Use of Brain Data from Clinical and Consumer-Directed Neurotechnological Devices."

### Health Economics

25. **Cost-Effectiveness**
    - Anderson (2004). "A cost-effectiveness analysis of brain-computer interface technology."
    - Neumann et al. (2016). "Cost-Effectiveness in Health and Medicine." Oxford University Press.

## Code Repositories to Study

### Open-Source Decoder Implementations
- **Neural_Decoding** (Glaser Lab): https://github.com/KordingLab/Neural_Decoding
- **LFADS** (Sussillo Lab): https://github.com/tensorflow/models/tree/master/research/lfads
- **AutoLFADS**: https://github.com/snel-repo/autolfads
- **Neural Data Transformers**: https://github.com/snel-repo/neural-data-transformers

### BCI Frameworks
- **MNE-Python**: https://mne.tools
- **Neo**: https://neo.readthedocs.io
- **Elephant**: https://elephant.readthedocs.io
- **BCI2000**: http://www.bci2000.org

### Deep Learning Libraries
- **PyTorch Lightning**: https://lightning.ai
- **TensorFlow**: https://www.tensorflow.org

## Online Courses & Tutorials

1. **Computational Neuroscience** - Coursera (University of Washington)
2. **Neural Data Science in Python** - Neuromatch Academy
3. **Deep Learning Specialization** - Coursera (Andrew Ng)
4. **Fast.ai** - Practical Deep Learning for Coders

## Conferences to Follow

- **Neural Information Processing Systems (NeurIPS)**
- **International Conference on Machine Learning (ICML)**
- **BCI Meeting (International BCI Society)**
- **Society for Neuroscience (SfN)**
- **Cosyne** (Computational and Systems Neuroscience)

## Reading Schedule (Month 1, Weeks 1-2)

### Week 1: Fundamentals
- Day 1-2: Papers 1-3 (Kalman, Wiener, LDA)
- Day 3-4: Papers 4-7 (ML approaches)
- Day 5-7: Papers 8-11 (Deep learning)

### Week 2: Advanced Topics
- Day 1-2: Papers 12-14 (Meta-learning)
- Day 3-4: Papers 16-17 (Clinical systems)
- Day 5: Papers 18-19 (Datasets)
- Day 6-7: Code repository exploration

---

**Note**: This is a living document. Add new references as you discover relevant work.

**Last Updated**: January 2026
