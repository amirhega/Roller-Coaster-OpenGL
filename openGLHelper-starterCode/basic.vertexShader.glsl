#version 150

in vec3 position;
in vec4 color;
in vec3 positionLeft, positionRight, positionUp, positionDown;
out vec4 col;


uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform int mode;
float eps = 0.00001f, smoothH;


void main()
{
  // compute the transformed and projected vertex position (into gl_Position) 
  // compute the vertex color (into col)
   
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0f);
        // col = color;
        col = vec4(1,1,1,1);
  
}

