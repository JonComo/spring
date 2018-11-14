import numpy as np
import pyglet
from pyglet.window import key
import json

width = 800
height = 400

dt = 0.1
time = 0
mousex = 0
mousey = 0
targx = 0
targy = 0
gravity = False
speedup = False
mouseDown = False
bs = []
ss = []
selected = None

# training
best = 0 # furthest right
brain = None

window = pyglet.window.Window(width, height)

def dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def avg_p():
    if len(bs) == 0:
        return {x: 0, y: 0}
    
    ax = 0
    ay = 0
    
    for b in bs:
        ax += b.x
        ay += b.y
        
    return {"x": ax/len(bs), "y": ay/len(bs)}

class Brain:
    def __init__(self, in_dim, out_dim, h_dim=30):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.rand_W()
        
    def rand_W(self):
        self.W1 = np.random.randn(self.h_dim, self.in_dim+9) * .4
        self.W2 = np.random.randn(self.out_dim, self.h_dim+1) * .4
        
    def process(self, springs):
        global time, targx, targy

        extra = np.zeros([9])
        extra[0] = 1
        extra[1] = np.sin(time)
        extra[2] = np.cos(time)
        extra[3] = np.sin(2*time)
        extra[4] = np.cos(2*time)

        # body angle
        a = bs[0]
        b = bs[-1]
        bang = np.arctan2(b.y - a.y, b.x - a.x)
        extra[5] = np.cos(bang)
        extra[6] = np.sin(bang)

        # encode dir to target relative to body angle
        cx = avg_p()['x']
        cy = avg_p()['y']
        ang = np.arctan2(cy-targy, cx-targx) - bang
        extra[5] = np.cos(ang)
        extra[6] = np.sin(ang)

        x = np.hstack((springs, extra))
        h1 = np.tanh(self.W1.dot(x))
        h1 = np.hstack((h1, np.array([1])))
        h2 = np.tanh(self.W2.dot(h1))
        return h2

class Box:
    def __init__(self, x, y, m=1, i=-1):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.m = m
        
        if i == -1:
            self.id = np.random.randint(1e10)
        else:
            self.id = i
        
    def data(self):
        return {"id": self.id, "x": self.x, "y": self.y, "m": self.m}
        
    def update(self):
        global gravity
        if gravity:
            self.ay -= self.m * 8

        #self.ax -= np.sign(self.vx) * self.vx*self.vx * .1
        #self.ay -= np.sign(self.vy) * self.vy*self.vy * .1
            
        self.vx += self.ax / self.m * dt
        self.vy += self.ay / self.m * dt
        self.x += self.vx
        self.y += self.vy
        self.ax = 0
        self.ay = 0
        
        if self.x < 0:
            self.x = 0
            self.vx *= -1
            self.vy *= 0.0
            
        if self.y < 0:
            self.y = 0
            self.vy *= -1
            self.vx *= 0.0
            
        if self.x > width:
            self.x = width
            self.vx *= -1
            self.vy *= 0.0
            
        if self.y > height:
            self.y = height
            self.vy *= -1
            self.vx *= 0.0
        
    def render(self):
        draw_rect(self.x + 5, self.y + 5, 10, 10)
        
class Spring:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.k = 40
        self.d = dist(self.a.x, self.a.y, self.b.x, self.b.y)
        self.id = np.random.randint(1e10)
        self.offset = 0 # control from brain
        self.diff = 0 # dist between a and b
        
    def data(self):
        return {"d": self.d, "a": self.a.id, "b": self.b.id}
        
    def update(self):
        curD = dist(self.a.x, self.a.y, self.b.x, self.b.y)
        self.diff = (self.d - curD)
        force = (self.diff + self.offset) * self.k
        ang = np.arctan2(self.a.y-self.b.y, self.a.x-self.b.x)
        
        c = np.cos(ang)
        s = np.sin(ang)

        # swimming
        vx = (self.a.vx + self.b.vx)/2.0
        vy = (self.a.vy + self.b.vy)/2.0
        ang2 = np.arctan2(vy, vx)

        diff = ang2-ang
        water = (1-np.cos(diff)*np.cos(diff)) * curD * .2 * np.sqrt(vx*vx + vy*vy)
        wx = water * np.cos(ang2+np.pi)
        wy = water * np.sin(ang2+np.pi)
        
        self.a.ax += (c * force + wx) * dt
        self.a.ay += (s * force + wy) * dt

        self.b.ax -= (c * force - wx) * dt
        self.b.ay -= (s * force - wy) * dt
        
    def render(self):
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, 
            ("v2f", (self.a.x, self.a.y, self.b.x, self.b.y)),
            ('c3B', (255, 255, 255, 255, 255, 255))
        )

def closest_box(x, y):
    global bs
    
    if len(bs) == 0:
        return None
    
    maxD = 1000000
    c = bs[0]
    for b in bs:
        d = dist(b.x, b.y, x, y)
        if d < maxD:
            maxD = d
            c = b
    return c

def box_with_id(i):
    global bs
    for b in bs:
        if b.id == i:
            return b
    print("box not found", i)
    return None

def export():
    global bs, ss
    
    data = {"boxes": [], "springs": []}
    
    for b in bs:
        data["boxes"].append(b.data())
        
    for s in ss:
        data["springs"].append(s.data())
        
    data_json = json.dumps(data)
    
    with open("save_file.txt", "w") as text_file:
        print(data_json, file=text_file)
    
def load():
    global bs, ss
    
    with open("save_file.txt", "r") as text_file:
        bs = []
        ss = []

        y = json.loads(text_file.read())

        for bdat in y["boxes"]:
            bs.append(Box(bdat["x"], bdat["y"], bdat["m"], bdat["id"]))

        for sdat in y["springs"]:
            a = box_with_id(sdat["a"])
            b = box_with_id(sdat["b"])

            spr = Spring(a, b)
            spr.d = sdat["d"]
            ss.append(spr)

@window.event
def on_key_press(symbol, modifiers):
    global bs, ss, selected
    
    if symbol == key.S:
        selected = closest_box(mousex, mousey)
        
@window.event
def on_key_release(symbol, modifiers):
    global bs, ss, selected, training, gravity, brain, best, speedup
    
    if symbol == key.C:
        bs = []
        ss = []
        brain = None
        best = 0
    elif symbol == key.G:
        gravity = not gravity
    elif symbol == key.F:
        speedup = not speedup
    elif symbol == key.T:
        print("train v1")
        train()
    elif symbol == key.Y:
        print("train finesse")
        train(True)
    elif symbol == key.B:
        bs.append(Box(mousex, mousey))
    elif symbol == key.E:
        export()
    elif symbol == key.L:
        load()
    elif symbol == key.S:
        b = closest_box(mousex, mousey)
        if selected and b and b is not selected:
            ss.append(Spring(selected, b))

@window.event        
def on_mouse_press(x, y, button, modifiers):
    global mouseDown, selected
    mouseDown = True
    selected = closest_box(x, y)

@window.event
def on_mouse_release(x, y, button, modifiers):
    global mouseDown
    mouseDown = False
    
@window.event
def on_mouse_motion(x, y, dx, dy):
    global mousex, mousey
    mousex = x
    mousey = y
    
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global mousex, mousey
    mousex = x
    mousey = y

def brain_update():
    global brain, bs, ss, time

    if brain is None:
        return
    
    x = np.array([s.diff/s.d for s in ss]) * .1
    #x = np.zeros_like(x)

    offs = brain.process(x)
    for i in range(len(offs)):
        ss[i].offset = offs[i] * ss[i].d * .2

def train2():
    global brain, time, ss, bs, best

    iters = 30

    load()
    if brain is None:
        brain = Brain(len(ss), len(ss))

    rs = np.zeros((iters))
    ds1 = []
    ds2 = []

    cW1 = brain.W1.copy()
    cW2 = brain.W2.copy()

    for j in range(iters):
        load()

        d1 = np.random.randn(cW1.shape[0], cW1.shape[1]) * .05
        d2 = np.random.randn(cW2.shape[0], cW2.shape[1]) * .05
        
        brain.W1 = cW1 + d1
        brain.W2 = cW2 + d2

        ds1.append(d1)
        ds2.append(d2)
            
        sx = avg_p()["x"]

        for i in range(250):
            time += dt
            brain_update()
            for s in ss:
                s.update()
            for b in bs:
                b.update()
                
        fx = avg_p()["x"]
        rs[j] = fx - sx
    
    rs -= np.mean(rs)
    rs /= np.std(rs)

    brain.W1 = cW1
    brain.W2 = cW2

    for j in range(iters):
        brain.W1 += ds1[j] * rs[j] * .05
        brain.W2 += ds2[j] * rs[j] * .05

def targ_dist():
    global targx, targy
    
    p = avg_p()
    dx = p['x']-targx
    dy = p['y']-targy

    return np.sqrt(dx*dx + dy*dy)

def draw_rect(x, y, dx, dy):
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x-dx, y, x-dx, y-dy, x, y-dy]))


def train(finesse=False):
    global brain, time, ss, bs, best, targx, targy

    load()
    if brain is None:
        brain = Brain(len(ss), len(ss))

    for j in range(10):
            
            lastW1 = brain.W1.copy()
            lastW2 = brain.W2.copy()

            if finesse:
                brain.W1 += np.random.randn(brain.W1.shape[0], brain.W1.shape[1]) * .01
                brain.W2 += np.random.randn(brain.W2.shape[0], brain.W2.shape[1]) * .01
            else:
                brain.rand_W()
            
            avg_r = 0
            avg_iters = 6
            for i in range(avg_iters):
                load()
                targx = np.random.random() * width
                targy = np.random.random() * height

                sd = targ_dist()

                for i in range(200):
                    time += dt
                    brain_update()
                    for s in ss:
                        s.update()
                    for b in bs:
                        b.update()
                        
                fd = targ_dist()

                avg_r += (sd - fd)/avg_iters

            if avg_r > best:
                best = avg_r
                print("new best", best)
            else:
                brain.W1 = lastW1
                brain.W2 = lastW2
        
def update(evt):
    global time, best, brain, targx, targy, mousex, mousey, speedup
    
    targx = mousex
    targy = mousey

    iters = 1
    if speedup:
        iters = 100

    for i in range(iters):
        time += dt
        brain_update()
        for s in ss:
            s.update()
        for b in bs:
            b.update()
            
@window.event
def on_close():
    print("closed")
    
@window.event
def on_draw():
    global mouseDown, mousex, mousey
    
    window.clear()
    
    if mouseDown and len(bs) > 0:
        selected.x = mousex
        selected.y = mousey
        selected.vx = 0
        selected.vy = 0
    
    for b in bs:
        b.render()
    for s in ss:
        s.render()
    
pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()