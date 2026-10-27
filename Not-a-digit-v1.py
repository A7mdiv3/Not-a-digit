import gradio as gr
import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from brainy import make_brain

brain_A = make_brain(10)
brain_A.load_state_dict(torch.load("baseline"))

brain_B = make_brain(11)
brain_B.load_state_dict(torch.load("baseline_B"))

def predict(drawing):
    if isinstance(drawing, dict):
        drawing=drawing["composite"]
    if drawing is None:
        return "Draw something first", ""
    
    pic=drawing.convert("RGBA")
    paper=Image.new("RGBA", pic.size, "white")
    paper.alpha_composite(pic)
    pen=ImageOps.invert(paper.convert("L"))
    pen=ImageOps.pad(pen.crop(pen.getbbox()), (100, 100))
    pen=pen.filter(ImageFilter.MaxFilter(9)).resize((20, 20))
    small=Image.new("L", (28, 28))
    small.paste(pen, (4, 4))
    numbers=torch.tensor(np.array(small), dtype=torch.float32) / 255
    numbers=numbers.reshape(1, 28, 28)
    chances=torch.softmax(brain_A(numbers), dim=1)

    if chances.max().item() < 0.9:
        answer_A="Not a digit"
    else:
        answer_A="It is a "+str(chances.argmax().item())

    answer=brain_B(numbers).argmax().item()
    if answer == 10:
        answer_B="Not a digit"
    else:
        answer_B="It is a " + str(answer)

    return answer_A, answer_B

web=gr.Interface(
    fn=predict,
    inputs=gr.Paint(type="pil", layers=False, canvas_size=(500, 500), brush=gr.Brush(default_size=10, colors=["#000000"], color_mode="fixed")),
    outputs=[gr.Textbox(label="Brain A"), gr.Textbox(label="Brain B")],
)
web.launch(share=True)
