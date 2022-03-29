AMIR HEGAZY

HW1;

    Program Features:
        Handles at least a 256x256 image for height field at interactive frame rates (window size of 1280x720). Height field transformations run smoothly.
        Renders the height field as points (key "1"), lines (key "2"), or solid triangles (key "3").
        Render a smoothened height field (key "4"), colored with a smoothened color.
        Render wireframe on top of solid triangles (key '5');
        Render as a perspective view, utilizing GL's depth buffer for hidden surface removal.
        Use input from the mouse to rotate the heightfield using OpenGLMatrix::Rotate.
        Use input from the mouse to move the heightfield using OpenGLMatrix::Translate.
        Use input from the mouse to change the dimensions of the heightfield using OpenGLMatrix::Scale.
        Color the vertices using a color that is linearly proportional to the height (with higher values being brighter; color cannot be all black).
        Takes images with color or greyscale and colors proportionaly

    Extra Credit:
        Wireframe (mode 5, key '5') on top of solid triangles in a constant color.
        Accept images with color (widthImg = heightImg).
            Colors image appropriately and gives a height field based off colors. If no color, uses greyscale color. 

    Needed to be brought to attention:
        Translate key had to be changed to the up arrow key
        Screenshot gets cut off (havent changed function, piazza instructor said it was okay to leave it after getting screenshots);
        