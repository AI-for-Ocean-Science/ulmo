from bokeh.models.widgets import TextInput
from bokeh.io import show

def my_text_input_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)

text_input = TextInput(value="default", title="Label:")
text_input.on_change("value", my_text_input_handler)

# This fails
show(text_input)