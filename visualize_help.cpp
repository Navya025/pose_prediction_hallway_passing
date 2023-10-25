#include <stdio.h>
#include <stdlib.h>

struct Joint {
    float x;
    float y;
    float z;
};

struct Bone {
    Joint startJoint;
    Joint endJoint;
};

struct Body {

    Joint joints[29];
    Bone bones[25];

};

void initialize_bones (Body human){

    //MAIN BODY 

    //initialize pelvis to naval bone
    human.bones[0].startJoint = human.joints[0];
    human.bones[0].endJoint = human.joints[1];

    //initialize naval to chest bone
    human.bones[1].startJoint = human.joints[1];
    human.bones[1].endJoint = human.joints[2];

    //initialize chest to neck bone
    human.bones[2].startJoint = human.joints[2];
    human.bones[2].endJoint = human.joints[3];

    //LEFT UPPER BODY 

    //initialize chest to left clavicle bone
    human.bones[3].startJoint = human.joints[2];
    human.bones[3].endJoint = human.joints[4];

    //initialize left clavicle to shoulder bone
    human.bones[4].startJoint = human.joints[4];
    human.bones[4].endJoint = human.joints[5];

    //initialize left shoulder to elbow bone
    human.bones[5].startJoint = human.joints[5];
    human.bones[5].endJoint = human.joints[6];

    //initialize left elbow to wrist bone
    human.bones[6].startJoint = human.joints[6];
    human.bones[6].endJoint = human.joints[7];

    //initialize left wrist to hand
    human.bones[7].startJoint = human.joints[7];
    human.bones[7].endJoint = human.joints[8];

    //initialize left hand to handtip bone
    human.bones[8].startJoint = human.joints[8];
    human.bones[8].endJoint = human.joints[9];

    //initialize left wrist to thumb bone
    human.bones[9].startJoint = human.joints[7];
    human.bones[9].endJoint = human.joints[10];

    //RIGHT UPPER BODY 

    //initialize chest to right clavicle bone
    human.bones[10].startJoint = human.joints[2];
    human.bones[10].endJoint = human.joints[11];

    //initialize right clavicle to shoulder bone
    human.bones[11].startJoint = human.joints[11];
    human.bones[11].endJoint = human.joints[12];

    //initialize right shoulder to elbow bone
    human.bones[12].startJoint = human.joints[12];
    human.bones[12].endJoint = human.joints[13];

    //initialize right elbow to wrist bone
    human.bones[13].startJoint = human.joints[13];
    human.bones[13].endJoint = human.joints[14];

    //initialize right wrist to hand
    human.bones[14].startJoint = human.joints[14];
    human.bones[14].endJoint = human.joints[15];

    //initialize right hand to handtip bone
    human.bones[15].startJoint = human.joints[15];
    human.bones[15].endJoint = human.joints[16];

    //initialize right wrist to thumb bone
    human.bones[16].startJoint = human.joints[14];
    human.bones[16].endJoint = human.joints[17];

    //LEFT LOWER BODY

    //initialize left pelvis to hip bone
    human.bones[17].startJoint = human.joints[0];
    human.bones[17].endJoint = human.joints[18];

    //initialize left hip to knee bone
    human.bones[18].startJoint = human.joints[18];
    human.bones[18].endJoint = human.joints[19];

    //initialize left knee to ankle bone
    human.bones[19].startJoint = human.joints[19];
    human.bones[19].endJoint = human.joints[20];

    //initialize left ankle to foot bone
    human.bones[20].startJoint = human.joints[20];
    human.bones[20].endJoint = human.joints[21];

    //RIGHT LOWER BODY

    //initialize right pelvis to hip bone
    human.bones[21].startJoint = human.joints[0];
    human.bones[21].endJoint = human.joints[22];

    //initialize right hip to knee bone
    human.bones[22].startJoint = human.joints[22];
    human.bones[22].endJoint = human.joints[23];

    //initialize right knee to ankle bone
    human.bones[23].startJoint = human.joints[23];
    human.bones[23].endJoint = human.joints[24];

    //initialize right ankle to foot bone
    human.bones[24].startJoint = human.joints[24];
    human.bones[24].endJoint = human.joints[25];

    //neck to head
    human.bones[25].startJoint = human.joints[3];
    human.bones[25].startJoint = human.joints[26];
}

void initialize_joints(Body human, float output[]) {
    int size = sizeof(output)/sizeof(output[0]);
    for (int i = 0; i < size-7; i+=7) {
        human.joints[i].x = output[i];
        human.joints[i].y = output[i+1];
        human.joints[i].z = output[i+2];
    }
}

void draw_body(Body human) {
    //draw all joints
    glBegin( GL_POINTS );
        for (int i = 0; i < 29; i++) {
            glVertex3f(human.joints[i].x, human.joints[i].y, human.joints[i].y);
        }
    glEnd();

    //draw all bones
    glBegin( GL_LINES);
        for (int i = 0; i < 29; i++) {
            glVertex3f(human.bones[i].startJoint.x, human.bones[i].startJoint.y, human.bones[i].startJoint.z);
            glVertex3f(human.bones[i].endJoint.x, human.bones[i].endJoint.y, human.bones[i].endJoint.z);
        }
    glEnd();
}