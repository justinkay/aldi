from helpers import SaveIO


class Distiller:

    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        # maybe register hooks

    def register_hooks(self):
        modules = ['roih', 'boxhead', 'boxpred']
        self.teacher_hooks = { f"{k}_io": SaveIO() for k in modules }
        self.student_hooks = { f"{k}_io": SaveIO() for k in modules }

    def get_losses(self):
