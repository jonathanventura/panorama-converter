from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from imageio import imread, imwrite
from scipy.special import cotdg
from skimage.transform import resize

vert = """
#version 330

layout(location = 0) in vec3 iPosition;

out vec3 texCoord;

uniform mat4 mvp;

void main()
{
    texCoord = iPosition;
    gl_Position = mvp * vec4(iPosition,1);
}
"""

frag = """
#version 330

uniform samplerCube sampler;

in vec3 texCoord;

out vec4 color;

void main()
{
    color = texture(sampler,texCoord);
    //color = vec4(texCoord*0.5+0.5,1);
    //color = vec4(normalize(texCoord),1);
}
"""

class Mesh:
    def __init__(self):
        self.vertices = []
        self.indices = []

    def initializeVertexBuffer(self):
        """
        Assign the triangular mesh data and the triplets of vertex indices that form the triangles (index data) to VBOs
        """
        self.vertexBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.indexBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def initializeVertexArray(self):
        """
        Creates the VAO to store the VBOs for the mesh data and the index data, 
        """
        self.vertexArrayObject = glGenVertexArrays(1)
        glBindVertexArray(self.vertexArrayObject)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        
        glBindVertexArray(0)

    def render(self):
        glBindVertexArray(self.vertexArrayObject)
        glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_SHORT, None)

class Cylinder(Mesh):
    def __init__(self,bottom,top,radius,nsegments=1024):
        thetarange = np.linspace(0,2*np.pi,nsegments)
        self.vertices = []
        for theta in thetarange:
            x = radius*np.cos(theta)
            z = radius*np.sin(theta)
            self.vertices.append(np.array([x,bottom,z]))
            self.vertices.append(np.array([x,top,z]))
        self.indices = []
        for i in range(len(self.vertices)):
            self.indices.append(np.array([i,(i+1)%len(self.vertices),(i+2)%len(self.vertices)]))
        self.vertices = np.stack(self.vertices,axis=0).astype(np.float32)
        self.indices = np.stack(self.indices,axis=0).astype(np.uint16)

class Cube(Mesh):
    def __init__(self,radius):
        a = 0.5#np.sqrt(3 * (radius**2))
        
        #ZPOS
        pos =  [(-a,a,a),(a,a,a),(-a,-a,a)]
        pos += [(-a,-a,a),(a,a,a),(a,-a,a)]
        #ZNEG
        pos += [(a,a,-a),(-a,a,-a),(-a,-a,-a)]
        pos += [(a,a,-a),(-a,-a,-a),(a,-a,-a)]
        #XNEG
        pos += [(-a,a,-a),(-a,a,a),(-a,-a,-a)]
        pos += [(-a,-a,-a),(-a,a,a),(-a,-a,a)]
        #XPOS
        pos += [(a,a,a),(a,a,-a),(a,-a,-a)]
        pos += [(a,a,a),(a,-a,-a),(a,-a,a)]
        #YPOS
        pos += [(-a,a,a),(-a,a,-a),(a,a,a)]
        pos += [(a,a,a),(-a,a,-a),(a,a,-a)]
        #YNEG
        pos += [(-a,-a,-a),(-a,-a,a),(a,-a,a)]
        pos += [(-a,-a,-a),(a,-a,a),(a,-a,-a)]

        self.vertices = np.array(pos,dtype='float')
        inds = []
        for i in range(0,36,3):
            inds.append([i,i+1,i+2])
        self.indices = np.array(inds,dtype='uint16')
        print(self.indices)

class Cube2(Mesh):
    def __init__(self,radius):
        r = radius
        self.vertices = np.array([
            [-r ,r ,-r],
            [-r,-r,-r],
            [r,-r,-r],

            [r,-r,-r],
            [r,r,-r],
            [-r,r,-r]
        ],dtype='float32')

        inds = []
        for i in range(0,len(self.vertices),3):
            inds.append([i,i+1,i+2])
        self.indices = np.array(inds,dtype='uint16')

class Renderer:
    def __init__(self,meshes,width,height,cubemappath):
        self.meshes = meshes
        self.width = width
        self.height = height
    
        shaderDict = {GL_VERTEX_SHADER: vert, GL_FRAGMENT_SHADER: frag}
        
        self.initializeShaders(shaderDict)
        
        # Set the dimensions of the viewport
        glViewport(0, 0, width, height)
        
        # Performs z-buffer testing
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)
        glDepthRange(0.0, 1.0)
        
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)

        # load cubemap
        self.loadCubeMap(cubemappath)
        
        # With all of our data defined, we can initialize our VBOs, FBO, and VAO to the OpenGL context to prepare for rendering
        self.initializeFramebufferObject()

        for mesh in self.meshes:
            mesh.initializeVertexBuffer()
            mesh.initializeVertexArray()
    
    def loadCubeMap(self,path):
        im = imread(path)[:,:,:3]
        size = 1024
        im = resize(im,(size*3,size*4),preserve_range=True).astype('uint8')
        print(im.shape,im.dtype)
        
        H,W = im.shape[:2]
        size = H//3

        right = im[size:2*size,2*size:3*size]
        left = im[size:2*size,:size]
        top = im[:size,size:2*size]
        bottom = im[2*size:3*size,size:2*size]
        back = im[size:2*size,3*size:4*size]
        front = im[size:2*size,size:2*size]
        
        front = front[::-1,::-1]
        left = left[::-1,::-1]
        right = right[::-1,::-1]
        back = back[::-1,::-1]

        images = [right,left,top,bottom,back,front]
        for i in range(6):
            imwrite(f'face{i}.jpg',images[i])
        
        self.texID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texID)
        for i in range(6):
            glTexImage2D( GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, np.flipud(images[i]) )
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)  

    def initializeShaders(self,shaderDict):
        """
        Compiles each shader defined in shaderDict, attaches them to a program object, and links them (i.e., creates executables that will be run on the vertex, geometry, and fragment processors on the GPU). This is more-or-less boilerplate.
        """
        shaderObjects = []
        self.shaderProgram = glCreateProgram()
        
        for shaderType, shaderString in shaderDict.items():
            shaderObjects.append(glCreateShader(shaderType))
            glShaderSource(shaderObjects[-1], shaderString)
            
            glCompileShader(shaderObjects[-1])
            status = glGetShaderiv(shaderObjects[-1], GL_COMPILE_STATUS)
            if status == GL_FALSE:
                if shaderType is GL_VERTEX_SHADER:
                    strShaderType = "vertex"
                elif shaderType is GL_GEOMETRY_SHADER:
                    strShaderType = "geometry"
                elif shaderType is GL_FRAGMENT_SHADER:
                    strShaderType = "fragment"
                raise RuntimeError("Compilation failure (" + strShaderType + " shader):\n" + glGetShaderInfoLog(shaderObjects[-1]).decode('utf-8'))
            
            glAttachShader(self.shaderProgram, shaderObjects[-1])
        
        glLinkProgram(self.shaderProgram)
        status = glGetProgramiv(self.shaderProgram, GL_LINK_STATUS)
        
        if status == GL_FALSE:
            raise RuntimeError("Link failure:\n" + glGetProgramInfoLog(self.shaderProgram).decode('utf-8'))
            
        for shader in shaderObjects:
            glDetachShader(self.shaderProgram, shader)
            glDeleteShader(shader)

    def configureShaders(self,mvp):
        mvpUnif = glGetUniformLocation(self.shaderProgram, "mvp")

        glUseProgram(self.shaderProgram)
        glUniformMatrix4fv(mvpUnif, 1, GL_TRUE, mvp)

        samplerUnif = glGetUniformLocation(self.shaderProgram, "sampler")
        glUniform1i(samplerUnif, 0)

        glUseProgram(0)

    def initializeFramebufferObject(self):
        """
        Create an FBO and assign a texture buffer to it for the purpose of offscreen rendering to the texture buffer
        """
        self.renderedTexture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.renderedTexture)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        self.depthRenderbuffer = glGenRenderbuffers(1)
        
        glBindRenderbuffer(GL_RENDERBUFFER, self.depthRenderbuffer)
        
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        
        self.framebufferObject = glGenFramebuffers(1)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.renderedTexture, 0)
        
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthRenderbuffer)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer binding failed, probably because your GPU does not support this FBO configuration.')
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def render(self,mvp):
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)

        self.configureShaders(mvp)

        glUseProgram(self.shaderProgram)

        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindTexture(GL_TEXTURE_CUBE_MAP,self.texID)
        glDepthMask(GL_FALSE)
        for mesh in self.meshes:
            mesh.render()
        glDepthMask(GL_TRUE)

        glUseProgram(0)

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        rendering = np.frombuffer(data, dtype = np.uint8).reshape(self.height, self.width, 3)

        return np.flipud(rendering)

def normalize(vec):
    return vec / np.linalg.norm(vec)

def lookAt(eye,center,up):
    F = center - eye
    f = normalize(F)
    s = np.cross(f,normalize(up))
    u = np.cross(normalize(s),f)

    M = np.eye(4)
    M[0,0:3] = s
    M[1,0:3] = u
    M[2,0:3] = -f
    
    T = np.eye(4)
    T[0:3,3] = -eye

    return M@T

def perspective(fovy,aspect,zNear,zFar):
    f = cotdg(fovy/2)
    M = np.zeros((4,4))
    M[0,0] = f/aspect
    M[1,1] = f
    M[2,2] = (zFar+zNear)/(zNear-zFar)
    M[2,3] = (2*zFar*zNear)/(zNear-zFar)
    M[3,2] = -1
    return M

