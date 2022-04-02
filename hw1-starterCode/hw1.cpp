/*
  CSCI 420 Computer Graphics, USC
  Assignment 1: Height Fields with Shaders.
  C++ starter code

  Student username: <type your USC username here>
*/

#include "basicPipelineProgram.h"
#include "texturePipelineProgram.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "openGLHeader.h"
#include "glutHeader.h"

#include <iostream>
#include <cstring>
#include <vector> //mine

#if defined(WIN32) || defined(_WIN32)
  #ifdef _DEBUG
    #pragma comment(lib, "glew32d.lib")
  #else
    #pragma comment(lib, "glew32.lib")
  #endif
#endif

#if defined(WIN32) || defined(_WIN32)
  char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
  char shaderBasePath[1024] = "../openGLHelper-starterCode";
#endif

using namespace std;

int mousePos[2]; // x,y coordinate of the mouse position

int leftMouseButton = 0; // 1 if pressed, 0 if not 
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0; // 1 if pressed, 0 if not

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// state of the world
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };
float scale = 0.2; // my global variable
int mode = 1;

int windowWidth = 1280;
int windowHeight = 720;
int imgWidth, imgHeight;
char windowTitle[512] = "CSCI 420 homework I";

ImageIO * heightmapImage;

//VBOs and their VAOs
GLuint triVertexBuffer, triColorVertexBuffer;
GLuint triVertexArray;
GLuint pointsVertexBuffer, pointsColorVertexBuffer; 
GLuint pointsVertexArray;
GLuint linesVertexBuffer, linesColorVertexBuffer; 
GLuint linesVertexArray;
GLuint wireFrameVertexBuffer, wireFrameColorVertexBuffer; 
GLuint wireFrameVertexArray;
GLuint trianglesVertexBuffer, trianglesColorVertexBuffer, trianglesVertexBuffer1, trianglesColorVertexBuffer1, trianglesVertexBuffer2, trianglesColorVertexBuffer2; 
GLuint sTrianglesVertexBuffer, sTrianglesColorVertexBuffer;
GLuint trianglesVertexArray;
GLuint sTrianglesVertexArray;
GLuint leftTrianglesVertexBuffer, leftTrianglesColorVertexBuffer,leftTrianglesVertexBuffer1,leftTrianglesVertexBuffer2; 
GLuint rightTrianglesVertexBuffer, rightTrianglesColorVertexBuffer; 
GLuint upTrianglesVertexBuffer, upTrianglesColorVertexBuffer; 
GLuint downTrianglesVertexBuffer, downTrianglesColorVertexBuffer; 
GLuint level1VertexBuffer, level1ColorVertexBuffer, binormalsVertexBuffer, normalsVertexBuffer;
GLuint level1VertexArray, level3VertexArray, level3VertexArray1, level3VertexArray2,normalsVertexArray, binormalsVertexArray;
GLuint level4VertexArray, level4VertexBuffer;
GLuint skyVertexArray, skyVertexBuffer;

GLuint groundTextHandle, skyTextHandle, railTextHandle;


vector<vector<float> > leftTrianglesVertices1,leftTrianglesVertices2,leftTrianglesVertices3;
vector<float> leftTriangleColors,leftTriangleColors1,leftTriangleColors2;
vector<vector<float> > rightTrianglesVertices1, rightTriangleColors;
vector<vector<float> > upTrianglesVertices1, upTriangleColors;
vector<vector<float> > downTrianglesVertices1, downTriangleColors;
vector<float> leftTrianglesVertices;
vector<float> rightTrianglesVertices;
vector<float> cTrianglesVertices;
vector<float> downTrianglesVertices;
vector<float> wireFrameVertices, wireFrameColors;
vector<float> level1Vertices, level1Color;
vector<float> xCam, yCam, zCam;
vector<float> bTemp;
vector<float> binormal, normals;
vector<float> groundVertices, groundPTex;
vector<float> skyVertices, skyPTex;


int hundreds = 0, tens = 0, ones = 0;
int stall = 0;
int countLook =100;
float uCamera = 1, uCamera2 = 0;
float ex=0,ey=0,ez=0,fx=0,fy=0,fz=0, ux=0.001,uy=0.001,uz=0.001, bx=0,by=0,bz=0;
float containerSize = 100;
float maxHeight = -100;
float beta = 0.03;


OpenGLMatrix matrix;
BasicPipelineProgram * pipelineProgram;
TexturePipelineProgram * texturePipelineProgram;

// represents one control point along the spline 
struct Point 
{
  double x;
  double y;
  double z;
};

vector<Point> spineNorms, spineTan, spinePoints, spineTanUnnorm;
// spline struct 
// contains how many control points the spline has, and an array of control points 
struct Spline 
{
  int numControlPoints;
  Point * points;
};

// the spline array 
Spline * splines;
// total number of splines 
int numSplines;
vector<Point> spline;

int loadSplines(char * argv) 
{
  char * cName = (char *) malloc(128 * sizeof(char));
  FILE * fileList;
  FILE * fileSpline;
  int iType, i = 0, j, iLength;

  // load the track file 
  fileList = fopen(argv, "r");
  if (fileList == NULL) 
  {
    printf ("can't open file\n");
    exit(1);
  }
  
  // stores the number of splines in a global variable 
  fscanf(fileList, "%d", &numSplines);

  splines = (Spline*) malloc(numSplines * sizeof(Spline));

  // reads through the spline files 
  for (j = 0; j < numSplines; j++) 
  {
    i = 0;
    fscanf(fileList, "%s", cName);
    fileSpline = fopen(cName, "r");

    if (fileSpline == NULL) 
    {
      printf ("can't open file\n");
      exit(1);
    }

    // gets length for spline file
    fscanf(fileSpline, "%d %d", &iLength, &iType);

    // allocate memory for all the points
    splines[j].points = (Point *)malloc(iLength * sizeof(Point));
    splines[j].numControlPoints = iLength;

    // saves the data to the struct
    while (fscanf(fileSpline, "%lf %lf %lf", 
	   &splines[j].points[i].x, 
	   &splines[j].points[i].y, 
	   &splines[j].points[i].z) != EOF) 
    {
      i++;
    }
  }

  free(cName);

  return 0;
}

int initTexture(const char * imageFilename, GLuint textureHandle)
{
  // read the texture image
  ImageIO img;
  ImageIO::fileFormatType imgFormat;
  ImageIO::errorType err = img.load(imageFilename, &imgFormat);

  if (err != ImageIO::OK) 
  {
    printf("Loading texture from %s failed.\n", imageFilename);
    return -1;
  }

  // check that the number of bytes is a multiple of 4
  if (img.getWidth() * img.getBytesPerPixel() % 4) 
  {
    printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
    return -1;
  }

  // allocate space for an array of pixels
  int width = img.getWidth();
  int height = img.getHeight();
  unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

  // fill the pixelsRGBA array with the image pixels
  memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
  for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++) 
    {
      // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
      pixelsRGBA[4 * (h * width + w) + 0] = 0; // red
      pixelsRGBA[4 * (h * width + w) + 1] = 0; // green
      pixelsRGBA[4 * (h * width + w) + 2] = 0; // blue
      pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

      // set the RGBA channels, based on the loaded image
      int numChannels = img.getBytesPerPixel();
      for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
        pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
    }

  // bind the texture
  glBindTexture(GL_TEXTURE_2D, textureHandle);

  // initialize the texture
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

  // generate the mipmaps for this texture
  glGenerateMipmap(GL_TEXTURE_2D);

  // set the texture parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // query support for anisotropic texture filtering
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
  printf("Max available anisotropic samples: %f\n", fLargest);
  // set anisotropic texture filtering
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

  // query for any errors
  GLenum errCode = glGetError();
  if (errCode != 0) 
  {
    printf("Texture initialization error. Error code: %d.\n", errCode);
    return -1;
  }
  
  // de-allocate the pixel array -- it is no longer needed
  delete [] pixelsRGBA;

  return 0;
}

// write a screenshot to the specified filename
void saveScreenshot(const char * filename)
{
  int scale = 2;
  int ww = windowWidth * scale;
  int hh = windowHeight * scale;
  unsigned char * screenshotData = new unsigned char[ww * hh * 3];
  glReadPixels(0, 0, ww, hh, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  unsigned char * screenshotData1 = new unsigned char[windowWidth * windowHeight * 3];
  for (int h = 0; h < windowHeight; h++) {
    for (int w = 0; w < windowWidth; w++) {
      int h1 = h * scale;
      int w1 = w * scale;
      screenshotData1[(h * windowWidth + w) * 3] = screenshotData[(h1 * ww + w1) * 3];
      screenshotData1[(h * windowWidth + w) * 3 + 1] = screenshotData[(h1 * ww + w1) * 3 + 1];
      screenshotData1[(h * windowWidth + w) * 3 + 2] = screenshotData[(h1 * ww + w1) * 3 + 2];
    }
  }

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData1);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
  delete [] screenshotData1;
}

void loadTexture(const char* imgName, GLuint& texHandle) {
  // create an integer handle for the texture 
  glGenTextures(1, &texHandle);
  cout << "tex: " << texHandle;
  int code = initTexture(imgName, texHandle); 
  if (code != 0) {
    cout << "Error loading the texture image.\n";
    exit(EXIT_FAILURE); 
  } 
}

//for multiple textures
void setTextureUnit(GLint unit) {
  GLuint program = texturePipelineProgram->GetProgramHandle();
  glActiveTexture(unit); // select texture unit affected by subsequent texture calls
  // get a handle to the “textureImage” shader variable
  GLint h_textureImage = glGetUniformLocation(program, "textureImage");
  // deem the shader variable “textureImage” to read from texture unit “unit”
  glUniform1i(h_textureImage, unit - GL_TEXTURE0);
}


void displayFunc()
{
  // render some stuff...
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.LoadIdentity();
  matrix.LookAt(ex, ey, ez, fx, fy, fz, ux, uy, uz); //changes by idle func
  
  //light
  pipelineProgram->Bind();
  float view[16];
  matrix.GetMatrix(view);

  GLuint program = pipelineProgram->GetProgramHandle();
  GLint h_viewLightDirection = glGetUniformLocation(program, "viewLightDirection");
  GLint ka = glGetUniformLocation(program, "ka");
  GLint kd = glGetUniformLocation(program, "kd");
  GLint ks = glGetUniformLocation(program, "ks");
  GLint alpha = glGetUniformLocation(program, "alpha");
  GLint La = glGetUniformLocation(program, "La");
  GLint Ld = glGetUniformLocation(program, "Ld");
  GLint Ls = glGetUniformLocation(program, "Ls");
  //silver
  float mat_ambient[] ={ 0.23125f, 0.23125f, 0.23125f, 1.0f };
  float mat_diffuse[] ={0.2775f, 0.2775f, 0.2775f, 1.0f };
  float mat_specular[] ={0.773911f, 0.773911f, 0.773911f, 1.0f };
  
  float shine =51.2f ;
  float light_ambient[] = {0.8f,0.8f,0.8f,1.0f};//{ 0.24725f, 0.1995f, 0.0745f, 1.0f };
  float light_diffuse[] = {0.75164f, 0.75164f, 0.75164f, 1.0f };
  float light_specular[] = {0.5f,0.5f,0.5,1.0f};//{0.628281f, 0.555802f, 0.366065f, 1.0f };
  // light direction
  float lightDirection[3] = { 0, 1, 0 }; // the “Sun” at noon 
  float viewLightDirection[3]; // light direction in the view space 
  // the following line is pseudo-code: 
  // viewLightDirection = (view * glm::vec4(lightDirection, 0.0)).xyz; 
  viewLightDirection[0] = (view[0] * lightDirection[0]) + (view[1] * lightDirection[1]) + (view[2] * lightDirection[2]);
  viewLightDirection[1] = (view[4] * lightDirection[0]) + (view[5] * lightDirection[1]) + (view[6] * lightDirection[2]);
  viewLightDirection[2] = (view[8] * lightDirection[0]) + (view[9] * lightDirection[1]) + (view[10] * lightDirection[2]);
  // upload viewLightDirection to the GPU 
  glUniform3fv(h_viewLightDirection, 1, viewLightDirection); 
  glUniform4fv(La, 1, light_ambient);
  glUniform4fv(Ld, 1, light_diffuse);
  glUniform4fv(Ls, 1, light_specular);
  glUniform4fv(ka, 1, mat_ambient);
  glUniform4fv(kd, 1, mat_diffuse);
  glUniform4fv(ks, 1, mat_specular);
  glUniform1f(alpha, shine);
  matrix.Translate(landTranslate[0],landTranslate[1], landTranslate[2]);
  matrix.Rotate(landRotate[0], 1,0,0);
  matrix.Rotate(landRotate[1], 0,1,0);
  matrix.Rotate(landRotate[2], 0,0,1);// landRotate[0], landRotate[1], landRotate[2]);
  matrix.Scale(landScale[0],landScale[1],landScale[2]);

  GLint h_normalMatrix = glGetUniformLocation(program, "normalMatrix");
  float m[16];
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetMatrix(m);

  GLboolean isRowMajor = GL_FALSE;
  glUniformMatrix4fv(h_normalMatrix, 1, isRowMajor, m);

  float p[16];
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.GetMatrix(p);
  
  // set variable
  pipelineProgram->SetModelViewMatrix(m);
  pipelineProgram->SetProjectionMatrix(p);

  // glBindVertexArray(level1VertexArray);
  // glDrawArrays(GL_LINE_STRIP, 0, level1Vertices.size()/3);
  

   glBindVertexArray(level3VertexArray1);
  glDrawArrays(GL_TRIANGLES, 0, leftTrianglesVertices.size()/3);
   glBindVertexArray(level3VertexArray2);
  glDrawArrays(GL_TRIANGLES, 0, rightTrianglesVertices.size()/3);

  // texture pipeline
  texturePipelineProgram->Bind();

  // set variable
  texturePipelineProgram->SetModelViewMatrix(m);
  texturePipelineProgram->SetProjectionMatrix(p);
  

  glBindTexture(GL_TEXTURE_2D, groundTextHandle);
  glBindVertexArray(level4VertexArray);
  glDrawArrays(GL_TRIANGLES, 0, groundVertices.size()/3);
  glBindVertexArray(0);


  glBindTexture(GL_TEXTURE_2D, skyTextHandle);
  glBindVertexArray(skyVertexArray);
  glDrawArrays(GL_TRIANGLES, 0, skyVertices.size()/3);
  glBindVertexArray(0);
  
  glBindTexture(GL_TEXTURE_2D, railTextHandle);
  glBindVertexArray(level3VertexArray);
  glDrawArrays(GL_TRIANGLES, 0, cTrianglesVertices.size()/3);
  glBindVertexArray(0);
  // glBindVertexArray(normalsVertexArray);
  // glDrawArrays(GL_LINES, 0, normals.size()/2);
  // glBindVertexArray(binormalsVertexArray);
  // glDrawArrays(GL_LINES, 0, binormal.size()/2);

  glutSwapBuffers();
}

float unitLength(float x, float y, float z) {
  return sqrt(pow(x,2) + pow(y,2)+ pow(z,2));
}

vector<float> unitCross(float a1,float a2,float a3, float b1, float b2, float b3) {
  // cout << "UNIT" << a1 << " " << a2 << " " << a3 << " " << b1 << " " << b2 << " " << b3 << endl;
  vector<float> c;
  c.push_back(a2*b3 - a3*b2);
  c.push_back(a3*b1 - a1*b3);
  c.push_back(a1*b2 - a2*b1);
  //check 
  float length = unitLength(c[0], c[1], c[2]);
  if(length != 0) {
    c[0] /= length;
    c[1] /= length;
    c[2] /= length;
  } else {
    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
  }
  return c;
}

vector<float> addVectors(float a1,float a2,float a3, float b1, float b2, float b3) {
  vector<float> c;
  c.push_back(a1+b1);
  c.push_back(a2+b2);
  c.push_back(a3+b3);
  return c;
}

float timeStep = 0.003;
float g = 9.8;
float uNew=0.01;

void idleFunc()
{   
   stall++;
  // do some stuff... 
  //Makes the screenshots
  // if(stall > 0) {
  //       if(ones > 9) {
  //           ones = 0;
  //           tens++;
  //       }
  //       if(tens > 9) {
  //           tens = 0;
  //           hundreds++;
  //       }
  //       if(hundreds <= 9) {
  //           string s = "images/" + to_string(hundreds) + to_string(tens)+ to_string(ones++) + ".jpg";
  //           char char_array[s.length() + 1];
  //           strcpy(char_array, s.c_str());
  //           saveScreenshot(char_array);
  //       }
  //       if(hundreds == 9 && tens == 9 && ones == 9) {
  //           string s = "images/" + to_string(hundreds) + to_string(tens)+ to_string(ones++) + ".jpg";
  //           char char_array[s.length() + 1];
  //           strcpy(char_array, s.c_str());
  //           saveScreenshot(char_array);
  //       }
  // }

  
  if(uCamera2 < spinePoints.size()-1) {
    ey = spinePoints[uCamera2].y+0.03 * spineNorms[uCamera2].y;
    ex = spinePoints[uCamera2].x+0.03 * spineNorms[uCamera2].x;
    ez = spinePoints[uCamera2].z+0.03 * spineNorms[uCamera2].z;
    
    float tanx = spineTan[uCamera2].x;//+0.03* spineNorms[uCamera2].x;
    float tany = spineTan[uCamera2].y;//+0.03* spineNorms[uCamera2].y;
    float tanz = spineTan[uCamera2].z;//+0.03* spineNorms[uCamera2].z;
    fx = tanx+ex;
    fy = tany+ey;
    fz = tanz+ez;
    //normal
    ux = spineNorms[uCamera2].x;
    uy = spineNorms[uCamera2].y;
    uz = spineNorms[uCamera2].z;
    uCamera2+= 1*uCamera;
  }
  // for example, here, you can save the screenshots to disk (to make the animation)
    // saveScreenshot("temp.jpg");
  // make the screen update 
  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.LoadIdentity();
  matrix.Perspective(60.0f, (float)w / (float)h, 0.01f, 1000.0f);
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);//GOOD PRACTICE
}

void mouseMotionDragFunc(int x, int y)
{
  // mouse has moved and one of the mouse buttons is pressed (dragging)

  // the change in mouse position since the last invocation of this function
  int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

  switch (controlState)
  {
    // translate the landscape
    case TRANSLATE:
      if (leftMouseButton)
      {
        // control x,y translation via the left mouse button
        landTranslate[0] += mousePosDelta[0] * 0.01f;
        landTranslate[1] -= mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z translation via the middle mouse button
        landTranslate[2] += mousePosDelta[1] * 0.01f;
      }
      break;

    // rotate the landscape
    case ROTATE:
      if (leftMouseButton)
      {
        // control x,y rotation via the left mouse button
        landRotate[0] += mousePosDelta[1];
        landRotate[1] += mousePosDelta[0];
      }
      if (middleMouseButton)
      {
        // control z rotation via the middle mouse button
        landRotate[2] += mousePosDelta[1];
      }
      break;

    // scale the landscape
    case SCALE:
      if (leftMouseButton)
      {
        // control x,y scaling via the left mouse button
        landScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
        landScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z scaling via the middle mouse button
        landScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
  // mouse has moved
  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

int keyUpPressed = 0;
//WHEN UP KEY IS PRESSED
void specialFunc(int key, int x, int y)
{
    if (key == GLUT_KEY_UP)
        keyUpPressed = 1;
}

//WHEN UP KEY IS REALSED
void ReleaseSpecialKeys(int key, int x, int y)
{
    if (key == GLUT_KEY_UP) {
        keyUpPressed = 0;
    }
}

void mouseButtonFunc(int button, int state, int x, int y)
{
  // a mouse button has has been pressed or depressed

  // keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
      leftMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_MIDDLE_BUTTON:
      middleMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_RIGHT_BUTTON:
      rightMouseButton = (state == GLUT_DOWN);
    break;
  }

  // keep track of whether CTRL and SHIFT keys are pressed
  switch (glutGetModifiers())
  {
    case GLUT_ACTIVE_SHIFT:
      controlState = SCALE;
    break;

    // if CTRL and SHIFT are not pressed, we are in rotate mode
    default:
      if(keyUpPressed) controlState = TRANSLATE;
      else controlState = ROTATE;
    break;
  }

  

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y)
{
    GLuint loc = glGetUniformLocation(pipelineProgram->GetProgramHandle(), "mode");
    GLuint num = 1;
  switch (key)
  {
    
    //points
    case '1':
        mode = 1;
        glUniform1ui(loc, 0); //use made vertex shader mode with uniform value
    break;

    //lines
    case '2':
        mode = 2;
        glUniform1i(loc, 0);
    break;
    
    //triangles
    case '3':
        mode = 3;
        glUniform1i(loc, 0);
    break;
    
    //smoothened triangles
    case '4':
        mode = 4;
        glUniform1i(loc, num); //set uniform value to mode =1 for written vertex shader
    break;

    //WIREFRAME MODE
    case '5':
        mode = 5;
        glUniform1i(loc, 0);
    break;

    //TRANSLATE,
    case 't':
      controlState = TRANSLATE;
    break;

    case 'r':
      uCamera2 = 0;
    break;

    case 'f':
      uCamera += 0.5;
    break;

    case 'g':
      if(uCamera >= 1.5)
        uCamera -= 0.5;
    break;

    case 27: // ESC key
      exit(0); // exit the program
    break;

    case ' ':
      cout << "You pressed the spacebar." << endl;
    break;

    case 'x':
      // take a screenshot
      saveScreenshot("screenshot.jpg");
    break;
  }
}

vector<float> calculateNormal(vector<float>& a, vector<float>& b ,vector<float>& c){
	vector<float> v1 = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
	vector<float> v2 = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
  vector<float> d;
  return unitCross(v1[0],v1[1],v1[2], v2[0],v2[1],v2[2]);
}

void addTriangleColor(int n1,int n2,int n3, int n4, int n5, int n6) {
  leftTriangleColors.push_back(n1);
  leftTriangleColors.push_back(n2);
  leftTriangleColors.push_back(n3);

  leftTriangleColors.push_back(n4);
  leftTriangleColors.push_back(n5);
  leftTriangleColors.push_back(n6);
}

void addTriangleColor1(vector<float> color) {
  //left rail for phong shading
  leftTriangleColors1.push_back(color[0]);
  leftTriangleColors1.push_back(color[1]);
  leftTriangleColors1.push_back(color[2]);
  
  leftTriangleColors1.push_back(color[0]);
  leftTriangleColors1.push_back(color[1]);
  leftTriangleColors1.push_back(color[2]);

  leftTriangleColors1.push_back(color[0]);
  leftTriangleColors1.push_back(color[1]);
  leftTriangleColors1.push_back(color[2]);  
}

void addTriangleColor2(vector<float> color) {
  //right rail for phong shading
  leftTriangleColors2.push_back(color[0]);
  leftTriangleColors2.push_back(color[1]);
  leftTriangleColors2.push_back(color[2]);
  
  leftTriangleColors2.push_back(color[0]);
  leftTriangleColors2.push_back(color[1]);
  leftTriangleColors2.push_back(color[2]);
  
  leftTriangleColors2.push_back(color[0]);
  leftTriangleColors2.push_back(color[1]);
  leftTriangleColors2.push_back(color[2]);  
  
}

double s = 0.5;
double M[4][4] = { {-s, 2-s, s-2, s} ,
        {2*s, s-3, 3-2*s, -s}, 
        {-s, 0, s, 0} ,
        {0, 1, 0, 0} };

void readSpline(char *file) {

  cout << file << endl;

  loadSplines(file);

  printf("Loaded %d spline(s).\n", numSplines);
  for(int i=0; i<numSplines; i++)
    printf("Num control points in spline %d: %d.\n", i, splines[i].numControlPoints);

  vector<double> p1;
  vector<double> p2;
  vector<double> p3;
  vector<double> p4;

  double C[4][3];
  Point coord;
  vector<float> v0, v1, v2, v3,v0left, v1left, v2left, v3left, v0right, v1right, v2right, v3right;


    //solely for getting max height
   for(int i = 0; i < numSplines; i++) {
    int hasFour = 0;
    for(int j = 0; j < splines[i].numControlPoints-3; j++) {
      for (int k = 0; k < 4; k++) { // fill out control matrix with P_k (k = 0 to 3) coefficients info of current spline (splines[i]), current point j
        C[k][0] = splines[i].points[j + k].x;
        C[k][1] = splines[i].points[j + k].y;
        C[k][2] = splines[i].points[j + k].z;
      }

      double B[4][3];
      for (int x = 0; x < 4; x++) { // each row of M1
          for (int y = 0; y < 3; y++) { // each column of M2
              B[x][y] = 0;
              for (int z = 0; z < 4; z++) { // each row of M2 - matches col of M1
                  B[x][y] += M[x][z] * C[z][y];
              }
          }
      }

      for (double u = 0.001; u <= 1.0; u += 0.001) {
        float xVar, xInd, yVar, yInd, zVar, zInd, xCord, yCord, zCord;
        xCord = pow(u, 3) * B[0][0] + pow(u, 2) * B[1][0] + pow(u, 1) * B[2][0] + B[3][0];
       yCord = pow(u, 3) * B[0][1] + pow(u, 2) * B[1][1] + pow(u, 1) * B[2][1] + B[3][1];
        zCord = pow(u, 3) * B[0][2] + pow(u, 2) * B[1][2] + pow(u, 1) * B[2][2] + B[3][2];
        Point sP;
        sP.x = xCord;
        sP.y = yCord;
        sP.z = zCord;
        if(yCord > maxHeight) maxHeight = yCord;
        // cout << yCord << endl;

        //Tangents
        xVar = ((3*pow(u, 2)) * B[0][0]) + ((2*u)* B[1][0]) + B[2][0];
        yVar = ((3*pow(u, 2)) * B[0][1]) + ((2*u)* B[1][1]) + B[2][1];
        zVar = ((3*pow(u, 2)) * B[0][2]) + ((2*u)* B[1][2]) + B[2][2];

        Point unNorm;
        unNorm.x = xVar;
        unNorm.y = yVar;
        unNorm.z = zVar;
        spineTanUnnorm.push_back(unNorm);
      }

    }
   }

    int numSkips =0;
      int count = numSkips;

  for(int i = 0; i < numSplines; i++) {
    int hasFour = 0;
    for(int j = 0; j < splines[i].numControlPoints-3; j++) {
      for (int k = 0; k < 4; k++) { // fill out control matrix with P_k (k = 0 to 3) coefficients info of current spline (splines[i]), current point j
        C[k][0] = splines[i].points[j + k].x;
        C[k][1] = splines[i].points[j + k].y;
        C[k][2] = splines[i].points[j + k].z;
      }

      double B[4][3];
      for (int x = 0; x < 4; x++) { // each row of M1
          for (int y = 0; y < 3; y++) { // each column of M2
              B[x][y] = 0;
              for (int z = 0; z < 4; z++) { // each row of M2 - matches col of M1
                  B[x][y] += M[x][z] * C[z][y];
              }
          }
      }

      coord.x = coord.y = coord.z = 0.0;
      for (double u = 0.001; u < 1.0; u += uNew) {
        float xVar, xInd, yVar, yInd, zVar, zInd, xCord, yCord, zCord;
        //position
        xCord = pow(u, 3) * B[0][0] + pow(u, 2) * B[1][0] + pow(u, 1) * B[2][0] + B[3][0];
        yCord = pow(u, 3) * B[0][1] + pow(u, 2) * B[1][1] + pow(u, 1) * B[2][1] + B[3][1];
        zCord = pow(u, 3) * B[0][2] + pow(u, 2) * B[1][2] + pow(u, 1) * B[2][2] + B[3][2];
        Point sP;
        sP.x = xCord;
        sP.y = yCord;
        sP.z = zCord;
        spinePoints.push_back(sP);
        level1Vertices.push_back(xCord);
        level1Vertices.push_back(yCord);
        level1Vertices.push_back(zCord);
        //physically accurate
        if(sP.y > maxHeight) maxHeight = sP.y;

        //Tangents
        xVar = ((3*pow(u, 2)) * B[0][0]) + ((2*u)* B[1][0]) + B[2][0];
        yVar = ((3*pow(u, 2)) * B[0][1]) + ((2*u)* B[1][1]) + B[2][1];
        zVar = ((3*pow(u, 2)) * B[0][2]) + ((2*u)* B[1][2]) + B[2][2];


        float tangentLength = sqrt(pow(xVar,2)+pow(yVar,2)+pow(zVar,2));
        //unNorm needed for it to bephysically accurate
        Point unNorm;
        unNorm.x = xVar;
        unNorm.y = yVar;
        unNorm.z = zVar;
        spineTanUnnorm.push_back(unNorm);
        if(tangentLength != 0) {
          xVar /= tangentLength;
          yVar /= tangentLength;
          zVar /= tangentLength;
        }
        Point tangent;
        tangent.x = xVar;
        tangent.y = yVar;
        tangent.z = zVar;
        xCam.push_back(xVar);
        yCam.push_back(yVar);
        zCam.push_back(zVar);
        xInd = xCam.size()-1;
        spineTan.push_back(tangent);

        //normal calc
        vector<float> temp;
        //if this is the first time
        if(v0.size() == 0) {
          temp = unitCross(tangent.x,tangent.y,tangent.z, 1,1,1);
        } else {
          temp = unitCross(bTemp[0], bTemp[1], bTemp[2], tangent.x,tangent.y,tangent.z);
        }
        Point norm;
        norm.x =temp[0];
        norm.y =temp[1];
        norm.z =temp[2];
         //adjust if all 0 to buffer it
        if(temp[0] == 0 && temp[1] == 0 && temp[2] == 0) {
          temp[0] = 0.0001;
          temp[1] = 0.00001;
          temp[2] = 0.0001;
        }
        
        spineNorms.push_back(norm);
        
        //binrmal calc
        bTemp = unitCross(tangent.x,tangent.y,tangent.z,norm.x,norm.y,norm.z);
        //adjust if all 0 to buffer it
         if(bTemp[0] == 0 && bTemp[1] == 0 && bTemp[2] == 0) {
          bTemp[0] = 0.0001;
          bTemp[1] = 0.00001;
          bTemp[2] = 0.0001;
        }
         cout << "Points: " << xCord << " " << yCord << " " << zCord << endl;
              cout << "Tanget: " << xVar << " " << yVar << " " << zVar << endl;
              cout << "Nornmal " << temp[0] << " " << temp[1] << " " << temp[2] << endl;
              cout << "BiNormal: " << bTemp[0] << " " << bTemp[1] << " " << bTemp[2] << endl;

        //for displaying normals
        normals.push_back(xCord);
        normals.push_back(yCord);
        normals.push_back(zCord);
        normals.push_back(xCord+temp[0]);
        normals.push_back(yCord+temp[1]);
        normals.push_back(zCord+temp[2]);

        //for displaying binormal
        binormal.push_back(xCord);
        binormal.push_back(yCord);
        binormal.push_back(zCord);
        binormal.push_back(xCord+bTemp[0]);
        binormal.push_back(yCord+bTemp[1]);
        binormal.push_back(zCord+bTemp[2]);

        //determine vertices of a side
        vector<float> nPlusb = addVectors(-norm.x,-norm.y,-norm.z, bTemp[0], bTemp[1], bTemp[2]);
        vector<float> nPlusb1 = addVectors(norm.x,norm.y,norm.z, bTemp[0], bTemp[1], bTemp[2]);
        vector<float> nPlusb2 = addVectors(norm.x,norm.y,norm.z, -bTemp[0], -bTemp[1], -bTemp[2]);
        vector<float> nPlusb3 = addVectors(-norm.x,-norm.y,-norm.z, -bTemp[0], -bTemp[1], -bTemp[2]);
        float alpha = 0.009;
        vector<float> v4 = addVectors(xCord, yCord, zCord, alpha*nPlusb[0], alpha*nPlusb[1], alpha*nPlusb[2]);              
        vector<float> v5 = addVectors(xCord, yCord, zCord, alpha*nPlusb1[0], alpha*nPlusb1[1], alpha*nPlusb1[2]);              
        vector<float> v6 = addVectors(xCord, yCord, zCord, alpha*nPlusb2[0], alpha*nPlusb2[1], alpha*nPlusb2[2]);              
        vector<float> v7 = addVectors(xCord, yCord, zCord, alpha*nPlusb3[0], alpha*nPlusb3[1], alpha*nPlusb3[2]);
        vector<float> v4right = v4;
        vector<float> v5right = v5;
        vector<float> v6right = v6;
        vector<float> v7right = v7;
        vector<float> v4left = v4;
        vector<float> v5left = v5;
        vector<float> v6left = v6;
        vector<float> v7left = v7;

        cout << xCord <<endl;
        //left rail
        v4left[0] -= beta*bTemp[0];
        v4left[1] -= beta*bTemp[1];
        v4left[2] -= beta*bTemp[2];
        v5left[0] -= beta*bTemp[0];
        v5left[1] -= beta*bTemp[1];
        v5left[2] -= beta*bTemp[2];

        //right rail
        v6right[0] += beta*bTemp[0];
        v6right[1] += beta*bTemp[1];
        v6right[2] += beta*bTemp[2];
        v7right[0] += beta*bTemp[0];
        v7right[1] += beta*bTemp[1];
        v7right[2] += beta*bTemp[2];
        cout << "BiNormal: " << v7[0] << " " << v7[1] << " " << v7[2] << endl;

        //not first one so we can draw triangles
        if(v0.size() != 0) {
        //crossbar (make thinner)
          v0[0] += 0.01*temp[0];
          v0[1] += 0.01*temp[1];
          v0[2] += 0.01*temp[2];
          v3[0] += 0.01*temp[0];
          v3[1] += 0.01*temp[1];
          v3[2] += 0.01*temp[2];
          if(count == 0) {
            //one side
            leftTrianglesVertices1.push_back(v6);
            leftTrianglesVertices1.push_back(v2);
            leftTrianglesVertices1.push_back(v1);
            addTriangleColor(0,1,0,0,1,0);

            leftTrianglesVertices1.push_back(v6);
            leftTrianglesVertices1.push_back(v5);
            leftTrianglesVertices1.push_back(v1);
            addTriangleColor(0,1,1,1,1,0);

            //one side
            leftTrianglesVertices1.push_back(v5);
            leftTrianglesVertices1.push_back(v1);
            leftTrianglesVertices1.push_back(v0);
            addTriangleColor(0,1,1,1,1,0);

            leftTrianglesVertices1.push_back(v5);
            leftTrianglesVertices1.push_back(v4);
            leftTrianglesVertices1.push_back(v0);
            addTriangleColor(0,1,1,1,1,0);

            //one side
            leftTrianglesVertices1.push_back(v7);
            leftTrianglesVertices1.push_back(v3);
            leftTrianglesVertices1.push_back(v0);
            addTriangleColor(0,1,1,1,1,0);

            leftTrianglesVertices1.push_back(v7);
            leftTrianglesVertices1.push_back(v4);
            leftTrianglesVertices1.push_back(v0);
            addTriangleColor(0,1,1,1,1,0);


            leftTrianglesVertices1.push_back(v6);
            leftTrianglesVertices1.push_back(v2);
            leftTrianglesVertices1.push_back(v3);
            addTriangleColor(0,1,1,1,1,0);

            leftTrianglesVertices1.push_back(v6);
            leftTrianglesVertices1.push_back(v7);
            leftTrianglesVertices1.push_back(v3);
            addTriangleColor(0,1,1,1,1,0);

            //front 
            leftTrianglesVertices1.push_back(v1);
            leftTrianglesVertices1.push_back(v2);
            leftTrianglesVertices1.push_back(v3);
            addTriangleColor(1,1,0,1,0,0);

            leftTrianglesVertices1.push_back(v1);
            leftTrianglesVertices1.push_back(v0);
            leftTrianglesVertices1.push_back(v3);
             addTriangleColor(1,1,1,0,0,0);


            count = numSkips;
          } else if(count > 0) {
            count--;
          }

          //LEFT RAIL
          leftTrianglesVertices2.push_back(v6left);
          leftTrianglesVertices2.push_back(v2left);
          leftTrianglesVertices2.push_back(v1left);
          vector<float> clr = calculateNormal(v6left,v2left,v1left);
          addTriangleColor1(clr);

          leftTrianglesVertices2.push_back(v6left);
          leftTrianglesVertices2.push_back(v5left);
          leftTrianglesVertices2.push_back(v1left);
          clr = calculateNormal(v1left,v5left,v6left);
          addTriangleColor1(clr);


          leftTrianglesVertices2.push_back(v5left);
          leftTrianglesVertices2.push_back(v1left);
          leftTrianglesVertices2.push_back(v0left);
          clr = calculateNormal(v5left,v1left, v0left);
          addTriangleColor1(clr);

          leftTrianglesVertices2.push_back(v5left);
          leftTrianglesVertices2.push_back(v4left);
          leftTrianglesVertices2.push_back(v0left);
          clr = calculateNormal(v0left,v4left,v5left);
          addTriangleColor1(clr);


          leftTrianglesVertices2.push_back(v7left);
          leftTrianglesVertices2.push_back(v3left);
          leftTrianglesVertices2.push_back(v0left);
          clr = calculateNormal(v7left,v3left,v0left);
          addTriangleColor1(clr);

          leftTrianglesVertices2.push_back(v7left);
          leftTrianglesVertices2.push_back(v4left);
          leftTrianglesVertices2.push_back(v0left);
          clr = calculateNormal(v0left,v4left,v7left);
          addTriangleColor1(clr);


          leftTrianglesVertices2.push_back(v6left);
          leftTrianglesVertices2.push_back(v2left);
          leftTrianglesVertices2.push_back(v3left);
          clr = calculateNormal(v6left,v2left,v3left);
          addTriangleColor1(clr);

          leftTrianglesVertices2.push_back(v6left);
          leftTrianglesVertices2.push_back(v7left);
          leftTrianglesVertices2.push_back(v3left);
          clr = calculateNormal(v3left,v7left,v6left);
          addTriangleColor1(clr);

          //Right
          //first traingle: v0 in this case is v4. triangle is v0,v1,v4
          leftTrianglesVertices3.push_back(v6right);
          leftTrianglesVertices3.push_back(v2right);
          leftTrianglesVertices3.push_back(v1right);
          clr = calculateNormal(v6right,v2right,v1right);
          addTriangleColor2(clr);

          leftTrianglesVertices3.push_back(v6right);
          leftTrianglesVertices3.push_back(v5right);
          leftTrianglesVertices3.push_back(v1right);
          clr = calculateNormal(v1right,v5right,v6right);
          addTriangleColor2(clr);


          leftTrianglesVertices3.push_back(v5right);
          leftTrianglesVertices3.push_back(v1right);
          leftTrianglesVertices3.push_back(v0right);
          clr = calculateNormal(v5right,v1right, v0right);
          addTriangleColor2(clr);

          leftTrianglesVertices3.push_back(v5right);
          leftTrianglesVertices3.push_back(v4right);
          leftTrianglesVertices3.push_back(v0right);
          clr = calculateNormal(v0right,v4right,v5right);
          addTriangleColor2(clr);


          leftTrianglesVertices3.push_back(v7right);
          leftTrianglesVertices3.push_back(v3right);
          leftTrianglesVertices3.push_back(v0right);
          clr = calculateNormal(v7right,v3right,v0right);
          addTriangleColor2(clr);

          leftTrianglesVertices3.push_back(v7right);
          leftTrianglesVertices3.push_back(v4right);
          leftTrianglesVertices3.push_back(v0right);
          clr = calculateNormal(v0right,v4right,v7left);
          addTriangleColor2(clr);


          leftTrianglesVertices3.push_back(v6right);
          leftTrianglesVertices3.push_back(v2right);
          leftTrianglesVertices3.push_back(v3right);
          clr = calculateNormal(v6right,v2right,v3left);
          addTriangleColor2(clr);

          leftTrianglesVertices3.push_back(v6right);
          leftTrianglesVertices3.push_back(v7right);
          leftTrianglesVertices3.push_back(v3right);
          clr = calculateNormal(v3right,v7right,v6right);
          addTriangleColor2(clr);
        }
        
        //update squares
        v0 = v4;
        v1 = v5;
        v2 = v6;
        v3 = v7;

        v0left = v4left;
        v1left = v5left;
        v2left = v6left;
        v3left = v7left;

        v0right = v4right;
        v1right = v5right;
        v2right = v6right;
        v3right = v7right;


       //physics equation to make it accurate
        float oldNew = uNew;
        
        uNew = timeStep * sqrt((2*g*(maxHeight - yCord)))/unitLength(xCord+unNorm.x,yCord+unNorm.y,zCord+unNorm.z);
        if(uNew == 0) uNew = 0.001;
        cout << "new: " << uNew << endl;
        if(uNew < 0.0004) uNew = 0.0004;

      }
    }
}
      //cause gaps in cross road
      numSkips = 2*12*4;
      count = numSkips;
      for(int i = 0; i < leftTrianglesVertices1.size(); i++) {
        if(count >0) {
        cTrianglesVertices.push_back(leftTrianglesVertices1[i][0]);
        cTrianglesVertices.push_back(leftTrianglesVertices1[i][1]);
        cTrianglesVertices.push_back(leftTrianglesVertices1[i][2]);
        
        } 
        count--;
        if(count <= -numSkips) count = numSkips;
      }

      for(int i = 0; i < leftTrianglesVertices2.size(); i++) {
        leftTrianglesVertices.push_back(leftTrianglesVertices2[i][0]);
        leftTrianglesVertices.push_back(leftTrianglesVertices2[i][1]);
        leftTrianglesVertices.push_back(leftTrianglesVertices2[i][2]);
      }

      for(int i = 0; i < leftTrianglesVertices3.size(); i++) {
        rightTrianglesVertices.push_back(leftTrianglesVertices3[i][0]);
        rightTrianglesVertices.push_back(leftTrianglesVertices3[i][1]);
        rightTrianglesVertices.push_back(leftTrianglesVertices3[i][2]);
      }
}



void initSky() {
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  //front
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  

  //right
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  //left
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(-containerSize);
  

  //top
  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  

  skyVertices.push_back(-containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(-containerSize);
  

  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);
  skyVertices.push_back(containerSize);

  //texture coord
  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(0);
  skyPTex.push_back(0);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(0);
  skyPTex.push_back(0);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(0);
  skyPTex.push_back(0);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(0);
  skyPTex.push_back(0);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

  skyPTex.push_back(0);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(1);

  skyPTex.push_back(1);
  skyPTex.push_back(0);

}

void initGround() {
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(-containerSize);

  groundVertices.push_back(containerSize);
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(-containerSize);

  groundVertices.push_back(-containerSize);
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(containerSize);

  groundVertices.push_back(containerSize);
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(-containerSize);

  groundVertices.push_back(-containerSize);
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(containerSize);

  groundVertices.push_back(containerSize);
  groundVertices.push_back(-containerSize);
  groundVertices.push_back(containerSize);

  groundPTex.push_back(0);
  groundPTex.push_back(1);

  groundPTex.push_back(1);
  groundPTex.push_back(1);

  groundPTex.push_back(0);
  groundPTex.push_back(0);

  groundPTex.push_back(1);
  groundPTex.push_back(1);

  groundPTex.push_back(0);
  groundPTex.push_back(0);

  groundPTex.push_back(1);
  groundPTex.push_back(0);

}


void initScene(int argc, char *argv[])
{
  
  //HW2
  glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
  loadTexture("wood.jpg", railTextHandle);
  loadTexture("sky.jpg", skyTextHandle);
  loadTexture("ground.jpg", groundTextHandle);
  
  readSpline(argv[1]);
  initGround();
  initSky();
  cout << level1Vertices.size();

    //level 1 spline
    glGenBuffers(1, &level1VertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, level1VertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(level1Vertices[0]) * level1Vertices.size(), &level1Vertices[0], GL_STATIC_DRAW);

    //cross section
    glGenBuffers(1, &leftTrianglesVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cTrianglesVertices[0]) * cTrianglesVertices.size(), cTrianglesVertices.data(), GL_STATIC_DRAW);

     glGenBuffers(1, &trianglesColorVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, (cTrianglesVertices.size()+leftTriangleColors.size()) * sizeof(float), cTrianglesVertices.data(), GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0,cTrianglesVertices.size() * sizeof(float),cTrianglesVertices.data());
    glBufferSubData(GL_ARRAY_BUFFER,cTrianglesVertices.size() * sizeof(float), leftTriangleColors.size() * sizeof(float), leftTriangleColors.data());

    //left rail
     glGenBuffers(1, &leftTrianglesVertexBuffer1);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(leftTrianglesVertices[0]) * leftTrianglesVertices.size(), leftTrianglesVertices.data(), GL_STATIC_DRAW);

     glGenBuffers(1, &trianglesColorVertexBuffer1);
    glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(leftTriangleColors1[0]) * leftTriangleColors1.size(), &leftTriangleColors1[0], GL_STATIC_DRAW);

    //right rail
     glGenBuffers(1, &leftTrianglesVertexBuffer2);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rightTrianglesVertices[0]) * rightTrianglesVertices.size(), rightTrianglesVertices.data(), GL_STATIC_DRAW);

     glGenBuffers(1, &trianglesColorVertexBuffer2);
    glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(leftTriangleColors2[0]) * leftTriangleColors2.size(), &leftTriangleColors2[0], GL_STATIC_DRAW);

    

    glGenBuffers(1, &normalsVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalsVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), &normals[0], GL_STATIC_DRAW);

    glGenBuffers(1, &binormalsVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, binormalsVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(binormal[0]) * binormal.size(), &binormal[0], GL_STATIC_DRAW);

    //textures
    glGenBuffers(1, &level4VertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, level4VertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, (groundVertices.size()+groundPTex.size()) * sizeof(float), groundVertices.data(), GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0,groundVertices.size() * sizeof(float),groundVertices.data());
    glBufferSubData(GL_ARRAY_BUFFER,groundVertices.size() * sizeof(float), groundPTex.size() * sizeof(float), groundPTex.data());

     //textures
    glGenBuffers(1, &skyVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, skyVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, (skyVertices.size()+skyPTex.size()) * sizeof(float), skyVertices.data(), GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0,skyVertices.size() * sizeof(float),skyVertices.data());
    glBufferSubData(GL_ARRAY_BUFFER,skyVertices.size() * sizeof(float), skyPTex.size() * sizeof(float), skyPTex.data());

    //BUILDS PIPELINE
    pipelineProgram = new BasicPipelineProgram;
    int ret = pipelineProgram->Init(shaderBasePath);
    if (ret != 0) abort();
    texturePipelineProgram = new TexturePipelineProgram;
    ret = texturePipelineProgram->Init(shaderBasePath);
    if (ret != 0) abort();

   

//This section binds vbos to VAO and sends variables to the shader

    //level 1 spline
    glGenVertexArrays(1, &level1VertexArray);
    glBindVertexArray(level1VertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, level1VertexBuffer);
    GLuint loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    glGenVertexArrays(1, &level3VertexArray);
    glBindVertexArray(level3VertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer);
    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
    glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer);
     loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "texCoord");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void*)(size_t)(cTrianglesVertices.size()*sizeof(float)));

    glGenVertexArrays(1, &level3VertexArray1);
    glBindVertexArray(level3VertexArray1);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer1);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
    glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer1);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "normal");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    glGenVertexArrays(1, &level3VertexArray2);
    glBindVertexArray(level3VertexArray2);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer2);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
    glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer2);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "normal");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    //normals
    glGenVertexArrays(1, &normalsVertexArray);
    glBindVertexArray(normalsVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, normalsVertexBuffer);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    glGenVertexArrays(1, &binormalsVertexArray);
    glBindVertexArray(binormalsVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, binormalsVertexBuffer);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    glGenVertexArrays(1, &level4VertexArray);
    glBindVertexArray(level4VertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, level4VertexBuffer);
    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "texCoord");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void*)(size_t)(groundVertices.size()*sizeof(float)));

     glGenVertexArrays(1, &skyVertexArray);
    glBindVertexArray(skyVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, skyVertexBuffer);
    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "texCoord");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void*)(size_t)(skyVertices.size()*sizeof(float)));

   


  glEnable(GL_DEPTH_TEST);

  std::cout << "GL error: " << glGetError() << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cout << "The arguments are incorrect." << endl;
    cout << "usage: ./hw1 <heightmap file>" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Initializing GLUT..." << endl;
  glutInit(&argc,argv);

  cout << "Initializing OpenGL..." << endl;

  #ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);  
  glutCreateWindow(windowTitle);

  cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(windowWidth - 1, windowHeight - 1);
  #endif

  // tells glut to use a particular display function to redraw 
  glutDisplayFunc(displayFunc);
  // perform animation inside idleFunc
  glutIdleFunc(idleFunc);
  // callback for mouse drags
  glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  glutMouseFunc(mouseButtonFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

  glutSpecialFunc(specialFunc);
  glutSpecialUpFunc(ReleaseSpecialKeys);

  // init glew
  #ifdef __APPLE__
    // nothing is needed on Apple
  #else
    // Windows, Linux
    GLint result = glewInit();
    if (result != GLEW_OK)
    {
      cout << "error: " << glewGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
    }
  #endif

  // do initialization
  initScene(argc, argv);

  // sink forever into the glut loop
  glutMainLoop();

  if (argc<2)
  {  
    printf ("usage: %s <trackfile>\n", argv[0]);
    exit(0);
  }
}


