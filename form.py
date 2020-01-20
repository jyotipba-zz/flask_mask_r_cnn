from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired,FileAllowed
from wtforms import  SubmitField
#from wtforms.validators import FileRequired
from werkzeug.utils import secure_filename

class ImageForm(FlaskForm):
    image = FileField('Image', validators=[FileRequired(), FileAllowed(['png', 'jpg', 'jpeg'])])
    submit = SubmitField('Upload image')
