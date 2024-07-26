import gradio as gr
from transformers import  AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, MBartForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoFeatureExtractor, AutoModelForAudioClassification
from ultralytics import YOLOWorld
import easyocr
from PIL import Image
import pandas as pd
from torch.utils.data import *
from torch.nn import *
from sklearn.metrics import *
import numpy as np
import torch
import os
import re


seed = 221
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def cleaning_text(text):
    text = text.strip()
    text = re.sub("r\s+", " ", text)
    return text


# Function Use
# TEXT

def change_text_dropdown_option(choice):
    return choice


def change_text_option(choice, language):
    visible_sen_emo = "Sentiment & Emotion" in choice
    visible_ner = "NER" in choice
    visible_sum = "Summarize" in choice
    visible_sen_emo_vn = ("Sentiment & Emotion" in choice) and ("English" in language)
    return{
        gen_text_context_length: gr.update(visible=visible_sum),
        sen_emo_output_1: gr.update(visible=visible_sen_emo),
        sen_emo_output_2: gr.update(visible=visible_sen_emo_vn),
        sen_emo_btn: gr.update(visible=visible_sen_emo),
        ner_output: gr.update(visible=visible_ner),
        ner_btn: gr.update(visible=visible_ner),
        sum_output: gr.update(visible=visible_sum),
        sum_btn: gr.update(visible=visible_sum),
        return_option:gr.update(visible= visible_sen_emo),
    }

def change_text_sen_emo_option(choice, language):
    visible_sen_emo_vn = ("Sentiment & Emotion" in choice) and ("English" in language)
    return{
        sen_emo_output_2: gr.update(visible=visible_sen_emo_vn)
    }



def token_length_compute(text):
    return len(cleaning_text(text).split(" "))

def ner_compute(text ,language):
    clean_input = cleaning_text(text)
    ner_model_name =""

    if(language == "English"):
        ner_model_name = "dslim/bert-base-NER"
    else:
        ner_model_name = "NlpHUST/ner-vietnamese-electra-base"
    
    
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

    ner_pipe = pipeline("ner", model = ner_model, tokenizer = ner_tokenizer)
    ner_output = ner_pipe(clean_input)
    return {"text": clean_input, "entities": ner_output} 

def sen_emo_compute(text, language, return_option):
    clean_input = cleaning_text(text)
    return_option = True if return_option == "True" else False

    if(language == "English"):
        sc_model_name = "ProsusAI/finbert"
    else:
        sc_model_name = "wonrax/phobert-base-vietnamese-sentiment"
    sc_tokenizer = AutoTokenizer.from_pretrained(sc_model_name)
    sc_model = AutoModelForSequenceClassification.from_pretrained(sc_model_name)
    sc_pipe = pipeline("text-classification", model = sc_model, tokenizer = sc_tokenizer, return_all_scores=return_option)
    sc_output = sc_pipe(clean_input)
    
    
    if(language == "English"):
        emo_model_name = "j-hartmann/emotion-english-distilroberta-base"
        emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_name)
        emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_name)

        emo_pipe = pipeline("text-classification", model = emo_model, tokenizer = emo_tokenizer, return_all_scores=return_option)
        emo_output = emo_pipe(clean_input)
    result1 = {}
    result2 = {}
    if(language == "English"):
        if(return_option == True):
            for item in sc_output[0]:
                result1[item["label"]] = item["score"]

            for item in emo_output[0]:
                result2[item["label"]] = item["score"]


            return result1, result2
        else:
            result1[sc_output[0]["label"]] = sc_output[0]["score"]
            result2[emo_output[0]["label"]] = emo_output[0]["score"]

            return result1, result2
           
    else:
        if(return_option == True):
            for item in sc_output[0]:
                result1[item["label"]] = item["score"]
            return result1, result2
        else:
            result1[sc_output[0]["label"]] = sc_output[0]["score"]
            return result1, result2
    
def sum_compute(text ,language, output_max_length):
    clean_input = cleaning_text(text)
    
    if(output_max_length > len(clean_input.split(" "))):
        output_max_length = len(clean_input.split(" "))
    output_min_length = int(output_max_length / 2)

    if(language == "English"):
        sum_model_name = "facebook/bart-large-cnn"
        sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)

        sum_pipe = pipeline("summarization", model = sum_model, tokenizer = sum_tokenizer, min_length = output_min_length ,max_length = output_max_length)
        sum_output = sum_pipe(clean_input)
        return sum_output[0]["summary_text"]
    else:
        sum_model_name = r"C:\Users\leduc\OneDrive\Desktop\NLP\Text-Summarization\model"
        sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
        sum_model = MBartForConditionalGeneration.from_pretrained(sum_model_name)
        sum_pipe = pipeline("summarization", model = sum_model, tokenizer = sum_tokenizer, min_length = output_min_length ,max_length = output_max_length)
        sum_output = sum_pipe(clean_input)
        return sum_output[0]["summary_text"]

# IMAGE
def change_slider(choice):
    visible_ob =  "Object Detection" in choice
    visible_cr =  "Context Retrieval" in choice
    visible_ocr =  "OCR" in choice
    return {
        conf_score: gr.update(visible=visible_ob),
        gen_context_length: gr.update(visible=visible_cr),
        OCR_output: gr.update(visible=visible_ocr),
        image_output: gr.update(visible=visible_ob),
        context_output: gr.update(visible=visible_cr),
        gen_context_beams: gr.update(visible=visible_cr),
        ob_btn: gr.update(visible=visible_ob),
        cr_btn: gr.update(visible=visible_cr),
        ocr_btn: gr.update(visible=visible_ocr),
        
    }

def detect_compute(img, conf):
    model = YOLOWorld(r"C:\Users\leduc\OneDrive\Desktop\ML_projects\THT-IMAGE\yolov8s-worldv2.pt")
    result = model(img, conf=conf)
    output_img = result[0].plot()
    img = Image.fromarray(output_img, mode="RGB")
    return img

def context_compute(img, max_length, beams):
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gen_kwargs = {"max_length": max_length, "num_beams": beams}

    inputs = processor(img, return_tensors = "pt").pixel_values
    output = model.generate(inputs, **gen_kwargs)
    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
    return result

def ocr_compute(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(img))
    text_result = ""
    for r in result:
        text_result = text_result + " " + r[1]
    return text_result

# AUDIO
def transcribe(audio):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-tiny"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    output = pipe(audio, generate_kwargs={"language": "english"})
    return output["text"]

def audio_cls_compute(audio):
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    pipe = pipeline("audio-classification", model=model, feature_extractor = feature_extractor, return_all_scores=True)
    output = pipe(audio)
    result = {}
    for item in output:
        result[item["label"]] = item["score"]
    return result



# Web App
with gr.Blocks() as app:
    gr.Markdown("Take Home Test App by NBTailee.")
    with gr.Tab("TEXT"):
        text_input = gr.TextArea(label="Input text")
        en_example = gr.Examples(["I met Jack in California at UC Berkerly university"], text_input)
        vn_example = gr.Examples(["Tao sẽ đấm vỡ mồm thằng Tuấn thu ngân ở Hải Phòng vì nó quá bố láo"], text_input)
        with gr.Accordion("Text Information Retrieval Options", open = True):
            text_option = gr.Dropdown(["English", "Vietnamese"], value="English" ,label="Languages")
            en_text_option = gr.CheckboxGroup(["Sentiment & Emotion", "NER", "Summarize"], info="Choose information retrieval options", label="Text Information Retrieval Options")
            return_option = gr.Dropdown(["True", "False"], value = "False" ,info="Choose to return all possible results for Sentiment & Emotion", label="Return Options", interactive=True ,visible=False)

            gen_text_context_length = gr.Slider(label="Max Length" ,info="Controll the max length of generation context", minimum = 30, maximum=150, step=1 ,interactive=True, visible=False)
            with gr.Row():
                sen_emo_btn = gr.Button("Compute Sentiment & Emotion", visible=False)
                ner_btn = gr.Button("Compute NER", visible=False)
                sum_btn = gr.Button("Compute Summarize", visible=False)
        tok_cnt_output = gr.Textbox(interactive=False, show_copy_button=True, label="Token Counts")
        sen_emo_output_1 = gr.Label(label="Sentiment", visible=False)
        sen_emo_output_2 = gr.Label(label="Emotion", visible=False)

        ner_output = gr.HighlightedText(interactive=False, label="ner", show_legend=False ,visible=False)
        sum_output = gr.TextArea(interactive=False, show_copy_button=True, label="Summarized Text", visible=False)


        # AI Controll
        text_input.change(fn=token_length_compute, inputs=[text_input], outputs=[tok_cnt_output])
        sen_emo_btn.click(fn=sen_emo_compute, inputs=[text_input, text_option, return_option], outputs=[sen_emo_output_1, sen_emo_output_2])
        ner_btn.click(fn=ner_compute, inputs=[text_input ,text_option], outputs=[ner_output])
        sum_btn.click(fn=sum_compute, inputs=[text_input ,text_option, gen_text_context_length], outputs=[sum_output])

        # Text option controll
        en_text_option.change(fn=change_text_option, inputs=[en_text_option, text_option], outputs=[gen_text_context_length, sen_emo_output_1, sen_emo_output_2, ner_output, sum_output, return_option, sen_emo_btn, ner_btn, sum_btn])
        text_option.change(fn=change_text_sen_emo_option, inputs=[en_text_option, text_option], outputs=[sen_emo_output_2])
    with gr.Tab("IMAGE"):
        image_input = gr.Image(label="Input Image")
        with gr.Accordion("Image Information Retrieval Options", open = True):
            model_option = gr.CheckboxGroup(["Object Detection", "OCR", "Context Retrieval"], info="Choose information retrieval options", label="options")
            conf_score = gr.Slider(label="Confidence Score" ,info="Controll the confidence score of model", minimum = 0.1, maximum=1, step=0.1 ,interactive=True, visible=False)
            gen_context_length = gr.Slider(label="Max Length" ,info="Controll the max length of generation context", minimum = 30, maximum=100, step=1 ,interactive=True, visible=False)
            gen_context_beams = gr.Slider(label="Generative Beams" ,info="Controll the quality of generation context", minimum = 1, maximum=6, step=1 ,interactive=True, visible=False)
            with gr.Row():
                ob_btn = gr.Button("Compute Detection", visible=False)
                cr_btn = gr.Button("Compute Context Retrieval", visible=False)
                ocr_btn = gr.Button("Compute OCR", visible=False)
        image_output = gr.Image(label="Output Image", show_download_button=True, interactive=False, visible=False)
        context_output = gr.TextArea(interactive=False, show_copy_button=True, label="Context of image", visible=False)
        OCR_output = gr.TextArea(interactive=False, show_copy_button=True, label="Extracted text from image", visible=False)

        # AI Controll
        ob_btn.click(fn=detect_compute, inputs=[image_input, conf_score], outputs=image_output)
        cr_btn.click(fn=context_compute, inputs=[image_input, gen_context_length, gen_context_beams], outputs=[context_output])
        ocr_btn.click(fn=ocr_compute, inputs=[image_input], outputs=[OCR_output])

        # Controlling visibility
        model_option.change(fn=change_slider, inputs=model_option, outputs=[conf_score, gen_context_length, OCR_output, context_output, gen_context_beams, ob_btn, cr_btn, ocr_btn, image_output])
    with gr.Tab("AUDIO"):
        
        audio_input = gr.Audio(label="Audio input", sources="upload", type="filepath")
        with gr.Row():
            audio_btn = gr.Button("Speech to text")
            audio_cls_btn = gr.Button("Sound Classification")                                                   
        audio_output = gr.TextArea(interactive=False, show_copy_button=True, label="Result")
        audio_cls_output = gr.Label(label="Sound")

        # AI Controll
        audio_btn.click( fn=transcribe ,inputs=audio_input, outputs=audio_output)
        audio_cls_btn.click(fn=audio_cls_compute, inputs=[audio_input], outputs=[audio_cls_output])
app.launch()