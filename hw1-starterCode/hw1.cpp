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
GLuint trianglesVertexBuffer, trianglesColorVertexBuffer; 
GLuint sTrianglesVertexBuffer, sTrianglesColorVertexBuffer;
GLuint trianglesVertexArray;
GLuint sTrianglesVertexArray;
GLuint leftTrianglesVertexBuffer, leftTrianglesColorVertexBuffer; 
GLuint rightTrianglesVertexBuffer, rightTrianglesColorVertexBuffer; 
GLuint upTrianglesVertexBuffer, upTrianglesColorVertexBuffer; 
GLuint downTrianglesVertexBuffer, downTrianglesColorVertexBuffer; 
GLuint level1VertexBuffer, level1ColorVertexBuffer, binormalsVertexBuffer, normalsVertexBuffer;
GLuint level1VertexArray, level3VertexArray, normalsVertexArray, binormalsVertexArray;
GLuint level4VertexArray, level4VertexBuffer;

GLuint groundTextHandle;


vector<vector<float> > leftTrianglesVertices1;
vector<float> leftTriangleColors;
vector<vector<float> > rightTrianglesVertices1, rightTriangleColors;
vector<vector<float> > upTrianglesVertices1, upTriangleColors;
vector<vector<float> > downTrianglesVertices1, downTriangleColors;
vector<float> leftTrianglesVertices;
vector<float> rightTrianglesVertices;
vector<float> upTrianglesVertices;
vector<float> downTrianglesVertices;
vector<float> wireFrameVertices, wireFrameColors;
vector<float> level1Vertices, level1Color;
vector<float> xCam, yCam, zCam;
vector<float> bTemp;
vector<float> binormal, normals;
vector<float> groundVertices, groundPTex;


int hundreds = 0, tens = 0, ones = 0;
int stall = 0;
int countLook =100;
int uCamera = 0, uCamera2 = 0;
float ex=0,ey=0,ez=0,fx=0,fy=0,fz=0, ux=0.001,uy=0.001,uz=0.001, bx=0,by=0,bz=0;


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

vector<Point> spineNorms, spineTan, spinePoints;
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
  unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
}

void displayFunc()
{
  // render some stuff...
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.LoadIdentity();
  // if(countLook < -100) countLook = 100;
  
  // int count = 100;
  // while(count  >0 ) {
    matrix.LookAt(ex, ey, ez, fx, fy, fz, ux, uy, uz);
    // matrix.LookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
    //  matrix.LookAt(13.5939, -0.0539935, 0.56969, 0, 0, 0, 0, 1, 0);
  // }
  matrix.Translate(landTranslate[0],landTranslate[1], landTranslate[2]);
  matrix.Rotate(landRotate[0], 1,0,0);
  matrix.Rotate(landRotate[1], 0,1,0);
  matrix.Rotate(landRotate[2], 0,0,1);// landRotate[0], landRotate[1], landRotate[2]);
  matrix.Scale(landScale[0],landScale[1],landScale[2]);


  float m[16];
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetMatrix(m);

  float p[16];
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.GetMatrix(p);
  
  // bind shader
  pipelineProgram->Bind();

  // set variable
  pipelineProgram->SetModelViewMatrix(m);
  pipelineProgram->SetProjectionMatrix(p);

  // glBindVertexArray(level1VertexArray);
  // glDrawArrays(GL_LINE_STRIP, 0, level1Vertices.size()/3);
  glBindVertexArray(level3VertexArray);
  glDrawArrays(GL_TRIANGLES, 0, rightTrianglesVertices.size()/3+leftTrianglesVertices.size()/3+upTrianglesVertices.size()/3+downTrianglesVertices.size()/3);

  // texturePipelineProgram->Bind();
  texturePipelineProgram->Bind();

  // set variable
  texturePipelineProgram->SetModelViewMatrix(m);
  texturePipelineProgram->SetProjectionMatrix(p);

  // // set variable
  // texturePipelineProgram->SetModelViewMatrix(m);
  // texturePipelineProgram->SetProjectionMatrix(p);
  //textures
    

    // loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "tc");
    // glEnableVertexAttribArray(loc);
    // glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void*)(size_t)(groundVertices.size()*sizeof(float)));
  glBindTexture(GL_TEXTURE_2D, groundTextHandle);
  glBindVertexArray(level4VertexArray);
  glDrawArrays(GL_TRIANGLES, 0, groundVertices.size()/3);
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
  //       if(hundreds < 3) {
  //           string s = "images/" + to_string(hundreds) + to_string(tens)+ to_string(ones++) + ".jpg";
  //           char char_array[s.length() + 1];
  //           strcpy(char_array, s.c_str());
  //           // saveScreenshot(char_array);
  //       }
  //       if(hundreds == 3 && tens == 0 && ones == 0) {
  //           string s = "images/" + to_string(hundreds) + to_string(tens)+ to_string(ones++) + ".jpg";
  //           char char_array[s.length() + 1];
  //           strcpy(char_array, s.c_str());
  //           // saveScreenshot(char_array);
  //       }
  // }

  if(uCamera < level1Vertices.size()-1) {
    ex = spinePoints[uCamera2].x+0.03 * spineNorms[uCamera2].x;
    ey = spinePoints[uCamera2].y+0.03 * spineNorms[uCamera2].y;
    ez = spinePoints[uCamera2].z+0.03 * spineNorms[uCamera2].z;
    
    float tanx = spineTan[uCamera2].x;//+0.03* spineNorms[uCamera2].x;
    float tany = spineTan[uCamera2].y;//+0.03* spineNorms[uCamera2].y;
    float tanz = spineTan[uCamera2].z;//+0.03* spineNorms[uCamera2].z;
    fx = tanx+ex;
    fy = tany+ey;
    fz = tanz+ez;

    ux = spineNorms[uCamera2].x;
    uy = spineNorms[uCamera2].y;
    uz = spineNorms[uCamera2].z;
    cout << "E: "<< ex << " " << ey << " " << ez << " " <<endl;
    cout << "F: "<< fx << " " << fy << " " << fz << " " <<endl;
    cout << "U: "<< ux << " " << uy << " " << uz << " " <<endl;
    uCamera2++;
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
  d.push_back(v1[1]*v2[2] - v1[2]*v2[1]);
  d.push_back(v1[2]*v2[1] - v1[1]*v2[2]);
  d.push_back(v1[1]*v2[1] - v1[1]*v2[1]);// = crossProduct(v1,v2);
	return d;
}

void addTriangleColor(vector<float> color) {
  leftTriangleColors.push_back(color[0]);
  leftTriangleColors.push_back(color[1]);
  leftTriangleColors.push_back(color[2]);
  leftTriangleColors.push_back(1.0f);

  leftTriangleColors.push_back(color[0]);
  leftTriangleColors.push_back(color[1]);
  leftTriangleColors.push_back(color[2]);
  leftTriangleColors.push_back(1.0f);

  leftTriangleColors.push_back(color[0]);
  leftTriangleColors.push_back(color[1]);
  leftTriangleColors.push_back(color[2]);  
  leftTriangleColors.push_back(1.0f);
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
  vector<float> v0, v1, v2, v3;

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
      for (double u = 0.001; u <= 1.0; u += 0.01) {
        float xVar, xInd, yVar, yInd, zVar, zInd, xCord, yCord, zCord;
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

        //Tangents
        xVar = ((3*pow(u, 2)) * B[0][0]) + ((2*u)* B[1][0]) + B[2][0];
        yVar = ((3*pow(u, 2)) * B[0][1]) + ((2*u)* B[1][1]) + B[2][1];
        zVar = ((3*pow(u, 2)) * B[0][2]) + ((2*u)* B[1][2]) + B[2][2];


        //check this, might be doing unit twice
        float tangentLength = sqrt(pow(xVar,2)+pow(yVar,2)+pow(zVar,2));
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

        //cross section rendering
        vector<float> temp;
        if(v0.size() == 0) {
          cout << "PISIDHSUDHSUDH";
          temp = unitCross(tangent.x,tangent.y,tangent.z, 1,1,1);
        } else {
          temp = unitCross(bTemp[0], bTemp[1], bTemp[2], tangent.x,tangent.y,tangent.z);
        }
        Point norm;
        norm.x =temp[0];
        norm.y =temp[1];
        norm.z =temp[2];
        spineNorms.push_back(norm);
        
        bTemp = unitCross(tangent.x,tangent.y,tangent.z,norm.x,norm.y,norm.z);
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

        binormal.push_back(xCord);
        binormal.push_back(yCord);
        binormal.push_back(zCord);
        binormal.push_back(xCord+bTemp[0]);
        binormal.push_back(yCord+bTemp[1]);
        binormal.push_back(zCord+bTemp[2]);

        vector<float> nPlusb = addVectors(-norm.x,-norm.y,-norm.z, bTemp[0], bTemp[1], bTemp[2]);
        vector<float> nPlusb1 = addVectors(norm.x,norm.y,norm.z, bTemp[0], bTemp[1], bTemp[2]);
        vector<float> nPlusb2 = addVectors(norm.x,norm.y,norm.z, -bTemp[0], -bTemp[1], -bTemp[2]);
        vector<float> nPlusb3 = addVectors(-norm.x,-norm.y,-norm.z, -bTemp[0], -bTemp[1], -bTemp[2]);
        float alpha = 0.01;
        vector<float> v4 = addVectors(xCord, yCord, zCord, alpha*nPlusb[0], alpha*nPlusb[1], alpha*nPlusb[2]);              
        vector<float> v5 = addVectors(xCord, yCord, zCord, alpha*nPlusb1[0], alpha*nPlusb1[1], alpha*nPlusb1[2]);              
        vector<float> v6 = addVectors(xCord, yCord, zCord, alpha*nPlusb2[0], alpha*nPlusb2[1], alpha*nPlusb2[2]);              
        vector<float> v7 = addVectors(xCord, yCord, zCord, alpha*nPlusb3[0], alpha*nPlusb3[1], alpha*nPlusb3[2]);

        if(v0.size() != 0) {
          //first traingle: v0 in this case is v4. triangle is v0,v1,v4
          leftTrianglesVertices1.push_back(v6);
          leftTrianglesVertices1.push_back(v2);
          leftTrianglesVertices1.push_back(v1);
          // vector<float> clr = calculateNormal(v6,v2,v1);
          // addTriangleColor(clr);

          leftTrianglesVertices1.push_back(v6);
          leftTrianglesVertices1.push_back(v5);
          leftTrianglesVertices1.push_back(v1);
          // addTriangleColor(clr);


          leftTrianglesVertices1.push_back(v5);
          leftTrianglesVertices1.push_back(v1);
          leftTrianglesVertices1.push_back(v0);
          // addTriangleColor(clr);

          leftTrianglesVertices1.push_back(v5);
          leftTrianglesVertices1.push_back(v4);
          leftTrianglesVertices1.push_back(v0);
          // addTriangleColor(clr);


          leftTrianglesVertices1.push_back(v7);
          leftTrianglesVertices1.push_back(v3);
          leftTrianglesVertices1.push_back(v0);
          // addTriangleColor(clr);

          leftTrianglesVertices1.push_back(v7);
          leftTrianglesVertices1.push_back(v4);
          leftTrianglesVertices1.push_back(v0);
          // addTriangleColor(clr);


          leftTrianglesVertices1.push_back(v6);
          leftTrianglesVertices1.push_back(v2);
          leftTrianglesVertices1.push_back(v3);
          // addTriangleColor(clr);

          leftTrianglesVertices1.push_back(v6);
          leftTrianglesVertices1.push_back(v7);
          leftTrianglesVertices1.push_back(v3);
          // addTriangleColor(clr);
        }
        
        v0 = v4;
        v1 = v5;
        v2 = v6;
        v3 = v7;
      }
    }
}
      for(int i = 0; i < leftTrianglesVertices1.size(); i++) {
        leftTrianglesVertices.push_back(leftTrianglesVertices1[i][0]);
        leftTrianglesVertices.push_back(leftTrianglesVertices1[i][1]);
        leftTrianglesVertices.push_back(leftTrianglesVertices1[i][2]);
      }
}

void loadTexture(const char* imgName, GLuint& texHandle) {
  // create an integer handle for the texture glGenTextures(1, &texHandle);
  int code = initTexture(imgName, texHandle); 
  if (code != 0) {
    cout << "Error loading the texture image.\n";
    exit(EXIT_FAILURE); 
  } 
}

void initGround() {
  groundVertices.push_back(-1000);
  groundVertices.push_back(-10);
  groundVertices.push_back(-1000);

  groundVertices.push_back(1000);
  groundVertices.push_back(-10);
  groundVertices.push_back(-1000);

  groundVertices.push_back(-1000);
  groundVertices.push_back(-10);
  groundVertices.push_back(1000);

  groundVertices.push_back(1000);
  groundVertices.push_back(-10);
  groundVertices.push_back(-1000);

  groundVertices.push_back(-1000);
  groundVertices.push_back(-10);
  groundVertices.push_back(1000);

  groundVertices.push_back(1000);
  groundVertices.push_back(-10);
  groundVertices.push_back(1000);

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
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  loadTexture("ground.jpg", groundTextHandle);
  
  readSpline(argv[1]);
     initGround();
  cout << level1Vertices.size();

    //level 1 spline
    glGenBuffers(1, &level1VertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, level1VertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(level1Vertices[0]) * level1Vertices.size(), &level1Vertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &leftTrianglesVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(leftTrianglesVertices[0]) * leftTrianglesVertices.size(), leftTrianglesVertices.data(), GL_STATIC_DRAW);

    //  glGenBuffers(1, &trianglesColorVertexBuffer);
    // glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(leftTriangleColors[0]) * leftTriangleColors.size(), &leftTriangleColors[0], GL_STATIC_DRAW);

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
    // glBufferData(GL_ARRAY_BUFFER, sizeof(groundVertices[0]) * groundVertices.size(), &groundVertices[0], GL_STATIC_DRAW);



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
    // glBindBuffer(GL_ARRAY_BUFFER, level1ColorVertexBuffer);
    // loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "color");
    // glEnableVertexAttribArray(loc);
    // glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    glGenVertexArrays(1, &level3VertexArray);
    glBindVertexArray(level3VertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, leftTrianglesVertexBuffer);
    loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
    // glBindBuffer(GL_ARRAY_BUFFER, trianglesColorVertexBuffer);
    // loc = glGetAttribLocation(pipelineProgram->GetProgramHandle(), "color");
    // glEnableVertexAttribArray(loc);
    // glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (const void *)0);

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
    // glBindTexture(GL_TEXTURE_2D, groundTextHandle);
    glBindVertexArray(level4VertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, level4VertexBuffer);
    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "position");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    //textures
    // glGenVertexArrays(1, &level4VertexArray);
    // glBindTexture(GL_TEXTURE_2D, groundTextHandle);
    // glBindVertexArray(level4VertexArray);
    // glBindBuffer(GL_ARRAY_BUFFER, level4VertexBuffer);
    // loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "position");
    // glEnableVertexAttribArray(loc);
    // glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

    loc = glGetAttribLocation(texturePipelineProgram->GetProgramHandle(), "texCoord");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void*)(size_t)(groundVertices.size()*sizeof(float)));


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


