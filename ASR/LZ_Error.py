#就是皮一下
class LZ_Error(Exception):
    def __init__(self,message):
        super().__init__(self)
        self.message=message
    def __str__(self):
        return self.message