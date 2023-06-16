import ipywidgets as widgets
from IPython.display import display
import openai
import requests
import json

# 定义交互界面的各个部分
openai_api_key_input = widgets.Text(
    placeholder='Enter your OpenAI API Key',
    description='OpenAI Key:',
    layout=widgets.Layout(width='400px')
)
huggingface_api_key_input = widgets.Text(
    placeholder='Enter your HuggingFace API Key',
    description='HF Key:',
    layout=widgets.Layout(width='400px')
)
init_apis_button = widgets.Button(
    description='Initialize APIs',
    button_style='info',
    layout=widgets.Layout(width='150px')
)
question_input = widgets.Text(
    placeholder='What objects are in the image at this URL?',
    description='Question:',
    layout=widgets.Layout(width='400px')
)
submit_button = widgets.Button(
    description='Submit',
    button_style='success',
    layout=widgets.Layout(width='100px')
)
output_area = widgets.Output()
response_output = widgets.Textarea(
    placeholder='AI Response will be displayed here.',
    description='AI Response:',
    layout=widgets.Layout(width='400px')
)

# 定义headers
headers = {}

# 展示所有部件
display(openai_api_key_input, huggingface_api_key_input, init_apis_button, question_input, submit_button, output_area, response_output)

def init_apis(button):
    # 这个函数将在点击"Initialize APIs"按钮后被调用，用于初始化OpenAI和Hugging Face API
    openai.api_key = openai_api_key_input.value
    headers["Authorization"] = f"Bearer {huggingface_api_key_input.value}"
    with output_area:
        print("APIs initialized successfully.")

def query(url):
    # 这个函数会向Hugging Face API发送图像URL，并获取图像中检测到的对象
    API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
    response = requests.get(url)
    response.raise_for_status()
    data = response.content
    api_response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(api_response.content.decode("utf-8"))

def  on_button_clicked(button):
    # 这个函数会在点击"Submit"按钮后被调用，用于处理图像对象检测任务
    with output_area:
        print("Processing the image object detection task...")
    function_descriptions = [
      {
          "name": "目标检测模型",
          "description": "Send an image URL to the Hugging Face API and get the detected objects in the image",
          "parameters": {
              "type": "object",
              "properties": {
                  "url": {
                      "type": "string",
                      "description": "The URL of the image to analyze",
                  }
              },
              "required": ["url"],
          },
        }
    ]
    user_query = question_input.value
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": user_query}],
        functions=function_descriptions,
        function_call="auto",
    )
    ai_response_message = response["choices"][0]["message"]
    url = eval(ai_response_message['function_call']['arguments']).get("url")
    function_response = query(url=url)
    second_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": user_query},
            ai_response_message,
            {
                "role": "function",
                "name": "query",
                "content": json.dumps(function_response),
            },
        ],
    )
    # 在函数内部，将AI的回答打印到新的输出文本区域
    response_output.value = second_response['choices'][0]['message']['content']

# 将函数与按钮的点击事件关联起来
init_apis_button.on_click(init_apis)
submit_button.on_click(on_button_clicked)
