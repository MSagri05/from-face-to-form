# From Face to Form
### An Emotion-Driven Recursive Generative Art System
**Manmeet Sagri · IAT 460 · Spring 2026**

🤗 **Try it live on Hugging Face:** https://huggingface.co/spaces/MSagri05/from-face-to-form

## About the app

From Face to Form takes a video, detects a face frame by frame, classifies the emotions using a pretrained AI model, and transforms those probabilities into two evolving generative art structures: a Julia fractal and an L-system tree. Both structures respond to the same emotional signal simultaneously, so as the emotional tone of the scene shifts, the visuals shift with it.


This is my final project for IAT 460: Computational Creativity at Simon Fraser University. The course prompt was to build a generative AI system, and throughout the semester we covered concepts like L-systems, generative grammars, fractals, and Markov models. I wanted to take what I learned in the labs and connect it to something more expressive, so instead of generating a static structure, I built a system that reads emotion from a video clip and uses that signal to drive the generative output in real time.

## Try It

Upload any short video clip that has a clearly visible face in it. A movie scene works great, but any clip where a face is readable works too. The system will detect the face, read the emotions across frames, and generate an evolving composite of the Julia fractal and L-system tree alongside the annotated source video.

🤗 **Live demo:** https://huggingface.co/spaces/MSagri05/from-face-to-form

## How to Run Locally

**1. Clone the repo**
```
git clone https://github.com/YOUR_USERNAME/from-face-to-form.git
cd from-face-to-form
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Run the app**
```
python3 app.py
```

The Gradio interface will open automatically in your browser. Upload a video clip and hit Generate Evolving Forms.

> Note: the first run will download the ViT emotion model from Hugging Face (~350MB). This only happens once and gets cached locally after that.

## System Pipeline

```
Video Input → Frame Sampling → Face Detection → Emotion Classification → Temporal Smoothing → Generative Output
```

**1. Frame Sampling**
The uploaded video is sampled at a fixed interval (every 0.2 seconds) using OpenCV. Processing every frame would be too slow at this scale, so sampling keeps things feasible while still capturing how emotion shifts across the scene.

**2. Face Detection**
Each sampled frame is passed through OpenCV's Haar Cascade classifier to find the largest face. The crop is extracted and passed to the emotion model.

**3. Emotion Classification**
The face crop is fed into a pretrained Vision Transformer (ViT) model (`trpakov/vit-face-expression` via Hugging Face) which outputs a probability distribution across 7 emotion categories: happy, sad, fear, angry, surprise, neutral, disgust.

**4. Temporal Smoothing**
Raw frame-to-frame emotion values can be noisy and unstable. Exponential moving average (EMA) smoothing is applied across the sequence so the generative output evolves gradually rather than flickering between frames.

**5. Generative Output**
The smoothed emotion probabilities simultaneously drive two systems:

Julia Fractal: the complex constant `c` is shifted by emotion values, zoom and turbulence are modulated, iteration depth varies, and color is blended from an emotion-coded palette. The full probability distribution influences the output, not just the dominant emotion.

L-System Tree: recursion depth, branching angle, step length, and production rules all respond to the active emotions. Fear produces erratic asymmetric branching, anger produces dense aggressive spread, sadness produces sparse drooping structures, and happy/neutral produces a balanced plant.

Both outputs are rendered side by side into a composite video at 5fps.

## Emotion to Parameter Mapping

| Emotion | Julia Fractal | L-System Tree |
|---------|--------------|---------------|
| Happy | Warm gold, expansive, high iteration depth | Tall dense tree, wider angle, longer step |
| Sad | Cool blue/violet, withdrawn, zoomed out | Sparse narrow tree, fewer iterations |
| Fear | Icy cyan, turbulent, irregular | Erratic asymmetric branching |
| Angry | Red/crimson, dense, high contrast | Aggressive spread, wide angle |
| Surprise | Hot pink, sudden expansion | Sharp angle shift, depth jump |
| Neutral | Grey, balanced, soft glow | Stable balanced plant |
| Disgust | Earthy green, warped | Irregular growth |

## Course Concepts Applied

**Julia Fractals (Week 2 Lab)**
The fractal generator is built on the escape-time algorithm introduced in Lab 2, where z → z² + c is iterated over the complex plane and points are colored by how long they take to escape. The extension here is making `c` a dynamic value driven by emotion probabilities rather than a fixed constant, so the shape of the fractal responds to how the scene feels.

**L-Systems (Week 3 Lab)**
The `build_lsystem_string` function directly mirrors the `create_l_system` function from the Week 3 lab, using the same axiom/rules/iterations structure and the same turtle command set (F, +, -, [, ]). The extension is making all of those parameters respond to the emotion signal, so the tree structure changes depending on what emotion is dominant.

**Generative Grammars (Week 3 Lab)**
Choosing different production rule sets for different emotional states is a direct application of parametric grammar design from the lab. Fear gets an irregular erratic grammar, anger gets a dense expansion grammar, sadness gets a minimal sparse grammar.

## Technical Challenges

**Image Model on Video Input**
The ViT model only accepts single images, not video. I had to build a frame sampling pipeline around it, extracting frames at fixed intervals, running detection and classification per frame, and reassembling the results into a temporal sequence.

**Video Autoplay and Codec Issues**
The generated output videos wouldn't play in the Gradio interface initially. Safari blocked autoplay (Chrome and Firefox worked fine), and OpenCV's default `mp4v` codec isn't browser-compatible. Switching to `avc1` (H.264) fixed the playback issue.

**L-System String Length**
The anger production rule expands very aggressively at higher recursion depths and can generate strings long enough to freeze rendering. A character cap of 8000 was added as a safety measure.

## AI Tools Used

Using AI tools was encouraged in this course, with the expectation that usage is properly referenced and that the student understands what was built.

**ChatGPT (OpenAI)**
I had no prior experience with the Hugging Face `transformers` library. I used ChatGPT to get oriented, understanding what `pipeline("image-classification", model=...)` returns, how to pass a PIL image to it, and how the output list is structured. The emotion processing pipeline (normalization, smoothing, blending, and parameter mapping) was then built and designed by me.

**Claude (Anthropic)**
The ViT model is an image classifier and can't process video directly. I used Claude to help me think through the frame-by-frame sampling approach, specifically how to extract frames at a fixed interval, feed each into the image model, and collect the outputs into a temporal sequence. The actual implementation and all the generative system logic were written by me. I also used Claude for the documentation pass on app.py.

## References

**Models**

Baltrusaitis, T. trpakov/vit-face-expression. Hugging Face. https://huggingface.co/trpakov/vit-face-expression

OpenCV Haar Cascade Face Detector. https://opencv.org

**Course Materials**

Week 2 Lab: Julia Fractals. IAT-ComputationalCreativity-Spring2026/Lab-2. https://github.com/IAT-ComputationalCreativity-Spring2026/Lab-2/blob/main/lab_fractals.ipynb

Week 3 Lab: L-Systems. IAT-ComputationalCreativity-Spring2026/Lab-3. https://colab.research.google.com/github/IAT-ComputationalCreativity-Spring2026/Lab-3/blob/main/l-systems.ipynb

Week 3 Lab: Generative Grammars. IAT-ComputationalCreativity-Spring2026/Lab-3. https://colab.research.google.com/github/IAT-ComputationalCreativity-Spring2026/Lab-3/blob/main/generative_grammars.ipynb

**Foundational Text**

Prusinkiewicz, P. and Lindenmayer, A. (1990). The Algorithmic Beauty of Plants. http://algorithmicbotany.org/papers/abop/abop.pdf

**Related Work**

Disney Research: Real Time Audience Emotion Tracking. https://www.cbc.ca/news/science/disney-ai-real-time-tracking-fvae-1.4233063

ImenTiv: Emotion AI in Filmmaking. https://imentiv.ai/emotion-ai-in-filmmaking/

**Video Clips Used for Testing**

Revolutionary Road (2008). YouTube clip. https://www.youtube.com/watch?v=n2lTpPptOWA

How to Lose a Guy in 10 Days (2003). YouTube clip. https://www.youtube.com/watch?v=WjSX_xAivY8

**AI Tools**

OpenAI. ChatGPT (GPT-4). Used for initial orientation with the Hugging Face Transformers library. https://openai.com

Anthropic. Claude (Claude Sonnet). Used for help thinking through the frame-by-frame video sampling approach and for the documentation pass on app.py. https://claude.ai
