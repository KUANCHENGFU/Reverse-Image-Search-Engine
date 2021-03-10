from wtforms import Form, FileField, SelectField, validators


class InputForm(Form):

    FileName = FileField(
        label='Image:',
        validators=[validators.InputRequired()]
        )
