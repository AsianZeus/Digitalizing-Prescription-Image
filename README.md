# Digitalizing-Prescription-Image
<p>PIRDS - Prescription Image Recognition and Digitalizing System is a OCR make with Tensorflow that digitalises images of Prescription of Handwritten Texts by Doctors.</p>

---

## Abstract
<p>PIRDS does the Digital transformation of hand-written prescription text using advance image processing techniques and deep learning methods. Image processing techniques helps to create images which are less noisy, and easily understandable for neural networks.

Once image with required configuration are obtained, they are fed to neural network model for training. The neural network model consists of, convolutional neural network for feature extraction, recurrent neural networks for dealing with character’s sequencing. We use connectionist temporal classification loss function which is required to be minimized to get good recognition of words from images.</p>

---

## Work Flow
1. The raw data are one-page scans, provided as a Images/PDF.
The first step is to anonymize the data. Hashes are calculated from document IDs, and a region of interest (ROI) is cut out of the document, which includes the handwriting, but which EXCLUDES any personal data, such as the physician’s signature, the date and place of decease, etc.
2. This yields smaller images than the originals, and there is no link from the images back to the original scans.
The second step is to clean the images. There is background text from the document template, and there are scan errors. We remove the background; we apply noise reduction and a slight blurring to close small gaps in the handwriting lines while retaining spaces between words.
3. The third step is to crop the image to the smallest size possible containing the handwriting.
The fourth step is to cut between the lines. Therefore, when the text has N lines, we end up with N image segments per original certificate.
4. We then apply a neural network (NN) to predict what is written; with a calculated confidence of how certain, the NN is of the correctness of the prediction.
Predictions that include unknown words require additional natural language processing (NLP) to map it to known words. Again, we calculate a confidence level.
5. To summarize, the solution for reading the handwriting is a combination of image processing, deep learning, and natural language processing.
