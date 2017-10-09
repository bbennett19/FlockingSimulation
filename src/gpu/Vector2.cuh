// Ben Bennett
// CIS 531
// Class definition for a 2D vector for use on the host and GPU.
// Note: I could have used a math lib for this, but I did not need much functionality
// and there were none that I could find installed on the server
#ifndef FLOCKING_VECTOR3_H
#define FLOCKING_VECTOR3_H

#include <iosfwd>
#include<GL/glx.h>


class Vector2 {
public:
    GLfloat x = 0.0;
    GLfloat y = 0.0;

    __device__ __host__ Vector2() {}
    __device__ __host__ Vector2(GLfloat xx, GLfloat yy);
    __device__ __host__ Vector2(const Vector2& a);
    __device__ __host__ GLfloat magnitudeSquared();
    __device__ __host__ GLfloat magnitude();
    __device__ __host__ Vector2 normalize();
    __device__ __host__ Vector2 operator+(const Vector2& b);
    __device__ __host__ Vector2 operator-(const Vector2& b);
    __device__ __host__ Vector2 operator*(const GLfloat& s);
    friend std::ostream&operator<<(std::ostream& stream, const Vector2& vector);
};


#endif