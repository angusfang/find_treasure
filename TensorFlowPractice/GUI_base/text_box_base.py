import pygame

class TextBoxBase():
    def __init__(self,t_surface:pygame.Surface):
        self.surface=t_surface
        self.font = pygame.font.Font('freesansbold.ttf', 12)
        self.text=''
        self.text_list=[]
        self.text_format= self.font.render(self.text, True, [0,0,0], [255,255,255])
        self.textRect = self.text_format.get_rect()
        self.textRect.topleft = (self.surface.get_width()-self.textRect.w,0)

    def use(self):
        self.text_format = self.font.render(self.text, True, [128,0,0], [128,255,255])
        self.textRect = self.text_format.get_rect()
        self.textRect.topleft = (self.surface.get_width() - self.textRect.w, 0)
        self.surface.blit(self.text_format, self.textRect)
        for index in range(len(self.text_list)):
            textI=self.text_list[index]
            self.text_format = self.font.render(textI, True, [128, 0, 0], [128, 255, 255])
            self.textRect = self.text_format.get_rect()
            self.textRect.topleft = (self.surface.get_width() - self.textRect.w, 1*index*self.textRect.h)
            self.surface.blit(self.text_format, self.textRect)


if __name__ == "__main__":
    pygame.init()
    surface = pygame.display.set_mode((1200, 700))

    surfRect = surface.get_rect()
    while True:
        surface.fill([255, 255, 255])
        text_box=TextBoxBase(surface)

        text_box.text_list.append('a')
        text_box.text_list.append('a')
        text_box.text_list.append('a')
        text_box.text_list.append('a')
        text_box.text_list.append('a')
        text_box.use()
        pygame.display.flip()