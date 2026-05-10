 
 
 - **i checked the model file to make sure it matched what we are describing but if someone could double/triple check that would be great. would be super embarassing and lose us hella points lmao**
  - most of our transformer encoder head is like the "Attention is All You Need" paper so we can reproduce similar results. the few differences are listed below. the major one being that in the paper the 6 layer network was the whole model, whereas our implementation stacks it on top of a pre-trained model to get better accuracy.
 - ESM was pretrained on millions of protein sequences using masked language modeling, so it gives context to the algorithm for the new proteins. 
 - the paper uses caching instead of backpropping, which takes out the expensive trainig part. the head trains a lot more quickly this way
  - we use multi-head self-attention, position-wise FFN, residual connections, and layer norm, all features from the paper. just at a smaller scale. this adds local refinement for our task. secondary structure has specific local signatures which can be modelled by these features
   - we are using 2 layers instead of the 6 in the paper because we are just making a small head on top of a large pre-trained model. 6 layers would overfit in our case.
    - output is 3 classes per residue because those are the three structures we are trying to predict