import gradio as gr
#test
from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import os
import uuid
import torch
from torch import autocast
import cv2
from matplotlib import pyplot as plt
from diffusers import DiffusionPipeline
from torchvision import transforms
from clipseg.models.clipseg import CLIPDensePredT

auth_token = os.environ.get("API_TOKEN") or True

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=auth_token,
).to(device)

model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval()
model.load_state_dict(torch.load('./clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)

transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.Resize((512, 512)),
])

def predict(radio, dict, word_mask, prompt=""):
    if(radio == "draw a mask above"):
        with autocast("cuda"):
            init_image = dict["image"].convert("RGB").resize((512, 512))
            mask = dict["mask"].convert("RGB").resize((512, 512))
    else:
        img = transform(dict["image"]).unsqueeze(0)
        word_masks = [word_mask]
        with torch.no_grad():
            preds = model(img.repeat(len(word_masks),1,1,1), word_masks)[0]
        init_image = dict['image'].convert('RGB').resize((512, 512))
        filename = f"{uuid.uuid4()}.png"
        plt.imsave(filename,torch.sigmoid(preds[0][0]))
        img2 = cv2.imread(filename)
        gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
        mask = Image.fromarray(np.uint8(bw_image)).convert('RGB')
        os.remove(filename)
    #with autocast("cuda"):
    output = pipe(prompt = prompt, image=init_image, mask_image=mask, strength=0.8)
    return output.images[0]

# examples = [[dict(image="init_image.png", mask="mask_image.png"), "A panda sitting on a bench"]]
css = '''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
'''
def swap_word_mask(radio_option):
    if(radio_option == "type what to mask below"):
        return gr.update(interactive=True, placeholder="A cat")
    else:
        return gr.update(interactive=False, placeholder="Disabled")

image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion Multi Inpainting
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Inpaint Stable Diffusion by either drawing a mask or typing what to replace
              </p>
            </div>
        """
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload").style(height=400)
            with gr.Box(elem_id="mask_radio").style(border=False):
                radio = gr.Radio(["draw a mask above", "type what to mask below"], value="draw a mask above", show_label=False, interactive=True).style(container=False)
                word_mask = gr.Textbox(label = "What to find in your image", interactive=False, elem_id="word_mask", placeholder="Disabled").style(container=False)
            prompt = gr.Textbox(label = 'Your prompt (what you want to add in place of what you are removing)')
            radio.change(fn=swap_word_mask, inputs=radio, outputs=word_mask,show_progress=False)
            radio.change(None, inputs=[], outputs=image_blocks, _js = """
            () => {
                css_style = document.styleSheets[document.styleSheets.length - 1]
                last_item = css_style.cssRules[css_style.cssRules.length - 1]
                last_item.style.display = ["flex", ""].includes(last_item.style.display) ? "none" : "flex";
            }""")
            btn = gr.Button("Run")
        with gr.Column():
            result = gr.Image(label="Result")
        btn.click(fn=predict, inputs=[radio, image, word_mask, prompt], outputs=result)
    gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://huggingface.co/CompVis" style="text-decoration: underline;" target="_blank">CompVis</a> and <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a> - Inpainting by <a href="https://github.com/nagolinc" style="text-decoration: underline;" target="_blank">nagolinc</a> and <a href="https://github.com/patil-suraj" style="text-decoration: underline;">patil-suraj</a>, inpainting with words by <a href="https://twitter.com/yvrjsharma/" style="text-decoration: underline;" target="_blank">@yvrjsharma</a> and <a href="https://twitter.com/1littlecoder" style="text-decoration: underline;">@1littlecoder</a> - Gradio Demo by 🤗 Hugging Face
                    </p>
                </div>
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
           """
        )
demo.launch()