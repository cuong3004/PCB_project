# from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import numpy as np
import cv2
import tkinter as tk
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Perform object detection on the image
def detect_objects(interpreter, image, input_size):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image (resize and normalize)
    resized_image = cv2.resize(image, (input_size, input_size))
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    return boxes, classes, scores

def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            # Scale bounding box coordinates to image size
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw rectangle and label
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"Class: {int(classes[i])}, Score: {scores[i]:.2f}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


window_width = 1280
window_height = 720
size_border = 7
title_hight = 100
split_width = window_width//2

background_color = "#F0F0F0"

def add_border(canvas):
    canvas.create_rectangle(0, 0, window_width, size_border, fill='blue', outline='blue')
    canvas.create_rectangle(0, window_height-size_border, window_width, window_height, fill='blue', outline='blue')
    canvas.create_rectangle(0, 0, size_border, window_height, fill='blue', outline='blue')
    canvas.create_rectangle(window_width-size_border, 0, window_width, window_height, fill='blue', outline='blue')

def add_text(canvas, text, text_position, font_size, font_color='white', font_thickness=2):
    canvas.create_text(text_position[0], text_position[1], text=text, font=('Helvetica', font_size, 'bold'), fill=font_color)

def insert_middle(canvas, pos_middle):
    canvas.create_rectangle(0, pos_middle, window_width, pos_middle + size_border, fill='white', outline='white')

def add_border_split(canvas):
    # add_border(canvas)
    canvas.create_rectangle(split_width, title_hight, split_width + size_border, window_height, fill='white', outline='white')

# def add_plot(canvas):
def clean(canvas):
    canvas.create_rectangle(0, 0, window_width, window_height, fill='black', outline='black')


class TkinterApp:
    def __init__(self, root):
        self.root = root

        self.canvas = tk.Canvas(self.root, width=window_width, height=window_height, bg=background_color, highlightthickness=0)
        self.canvas.pack()

        self.create_gui_normal_mode()


    def create_gui_normal_mode(self):
        
        self.insert_label()
        
        add_border(self.canvas)
        # add_border_split(self.canvas)

        # insert_middle(self.canvas, title_hight)
        add_text(self.canvas, "PCB Classification System", (window_width//2, title_hight//2), 23, "black")
        
        self.insert_box()
        self.insert_bolder_frame()
        

    def insert_label(self):
        font_tuple = ('Helvetica', 15, 'bold')
        font_tuple_legend = ('Helvetica', 20, 'bold')
        pos_button = (900, 200)
        size_button = (50,50)
        fgs = [
            "#FFA39E",
            "#D4380D",
            "#FFC069",
            "#AD8B00",
            "#D3F261",
            "#389E0D",
            "#5CDBD3"
        ]
        
        texts = [
            "IC",
            "Capacitor",
            "SMD",
            "Resistor",
            "Inductor",
            "Transistor",
            "Other"
        ]
        
        label_legend = tk.Label(self.root, text="Compoment       Quantity", fg = "black", borderwidth=1,
                            bg = background_color, font=font_tuple)
        label_legend.place(x=pos_button[0], y=pos_button[1]-50)
        
        for i in range(3):
            for j in range(7):
                # print(i,j)
                pad2 = 0
                if i == 0:
                    text = ' ❑ '
                    _font_tuple = font_tuple_legend
                elif i == 1:
                    text = texts[j]
                    _font_tuple = font_tuple
                    
                elif i == 2:
                    text = ':       0'
                    _font_tuple = font_tuple
                    pad2 = 50
                    # pad2 = 0
                    
                label_legend = tk.Label(self.root, text=text, fg =fgs[j], borderwidth=1,
                            bg = background_color, font=_font_tuple)
                label_legend.place(x=pos_button[0]+i*size_button[0]+pad2, y=pos_button[1]+j*size_button[1])
                setattr(self, f"label_legend_{i}_{j}", label_legend)
                
                
        
    def insert_bolder_frame(self):
        frame_width = 512
        frame_height = frame_width//1.333
        frame_pos = (100,140)
        # box_width = 90
        # box_height = 70
        # pad_2_box = 500
        self.canvas.create_rectangle(*frame_pos, frame_pos[0]+frame_width, frame_pos[1]+frame_height, outline='white', width=3, fill="gray" )
        
        # self.
    
    def insert_box(self):
        pad_box = 100
        box_width = 90
        box_height = 70
        pad_2_box = 420
        height_pos = window_height//1.3
        font_tuple = ('Helvetica', 15, 'bold')
        pad_label_box = (-60,40)
        
        self.canvas.create_rectangle(pad_box, height_pos, pad_box+box_width, height_pos+box_height, outline='white', width=3, fill="#AFABAA" ) #fill='white'
        add_text(self.canvas, "Box 1", (pad_box+box_width//2, height_pos+box_height//2), 15, "black")
        self.box_1_label = tk.Label(self.root, text="Quantity: 0", fg ="black", borderwidth=1,
                            bg = background_color, font=font_tuple)
        self.box_1_label.place(x=pad_box+box_width//2+pad_label_box[0], y=height_pos+box_height//2+pad_label_box[1])
        
        
        _pad_box = pad_box + pad_2_box
        self.canvas.create_rectangle(_pad_box, height_pos, _pad_box+box_width, height_pos+box_height, outline='white', width=3, fill="#A6807E") #fill='white'
        add_text(self.canvas, "Box 2", (_pad_box+box_width//2, height_pos+box_height//2), 15, "black")
        self.box_2_label = tk.Label(self.root, text="Quantity: 0", fg ="black", borderwidth=1,
                            bg = background_color, font=font_tuple)
        self.box_2_label.place(x=_pad_box+box_width//2+pad_label_box[0], y=height_pos+box_height//2+pad_label_box[1])


    def check_state(self):
        print(self.checkbox_var.get())
        if self.checkbox_var.get():
            print("Checkbutton is checked")
        else:
            print("Checkbutton is unchecked")

    def toggle_choice_widget(self):
        if self.show_choice_var.get():
            # Show the choice widget
            self.show_choice()
        else:
            # Hide the choice widget
            self.hide_choice()

    def plot_matplotlib(self):
        # Matplotlib code
        x_data = [1, 2, 3, 4, 5]
        y_data = [10, 12, 5, 8, 15]

        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, marker='o')
        ax.set_title('Matplotlib Plot')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Embed Matplotlib figure in Tkinter Canvas
        canvas_matplotlib = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_matplotlib.draw()
        canvas_matplotlib.get_tk_widget().place(x=0,y=150)

    def run(self):
        self.root.mainloop()



if __name__ == "__main__":
    root = tk.Tk()
    # root.attributes('-fullscreen', True)
    root.geometry(f"{window_width}x{window_height}")
    root.title("SmartMirror")
    root.configure(background="black")

    app = TkinterApp(root)
    app.run()


