import glob

html_string_start = """
{% extends "layout.html" %}
{% block content %}
    <form class="form-signin" method=post enctype=multipart/form-data>
		<h1 class="h2 mb-3">Sewer Defect Segmentation</h1>
"""

html_string_end = """
    </form>
{% endblock %}
"""

def get_video_html(inp, out):
    video_html = """
		<video preload="auto" autoplay controls loop style="width: 100% !important; height: auto !important;">
			<source src="../{inp_name}" type="video/mp4">
			Your browser does not support the video tag.
		</video>
		<video preload="auto" autoplay controls loop style="width: 100% !important; height: auto !important;">
			<source src="../{out_name}" type="video/mp4">
			Your browser does not support the video tag.
		</video>
    """
    return video_html.format(inp_name=inp, out_name=out)


def get_picture_html(inp, out):
    image_html = """
        <p> {out_name} </p>
        <img id="result-input" src= "../{inp_name}"/>
        <img id="result-output" src= "../{out_name}"/>
    """
    return image_html.format(inp_name=inp, out_name=out)


def get_count_html(category, count):
    count_html = """<li> {category_name} : {count_} </li>"""
    return count_html.format(category_name=category, count_=count)


def get_value_count(image_class_dict):
    count_dic = {}
    for category in image_class_dict.values():
        if category in count_dic.keys():
            count_dic[category] = count_dic[category] + 1
        else:
            count_dic[category] = 1
    return count_dic


def generate_html(inp=None, out=None, img_list=None):
    picture_html = ""

    if img_list is not None:
        for value in img_list:
            inp, out = value.split(':')
            picture_html += get_picture_html(inp, out)
    if inp is not None and out is not None and img_list is None:
        if inp.split('.')[1] == 'jpg' or inp.split('.')[1] == 'png':
            picture_html += get_picture_html(inp, out)
        elif inp.split('.')[1] == 'mp4':
            picture_html += get_video_html(inp, out)

    file_content = html_string_start + picture_html + html_string_end

    with open('templates/results.html', 'w') as f:
        f.write(file_content)
