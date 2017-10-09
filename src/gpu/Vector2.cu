// Ben Bennett
// CIS 531
// Implementation of the Vector2 class for use on the host and GPU
#include "Vector2.cuh"
#include <math.h>
#include <iostream>

__device__ __host__
Vector2::Vector2(GLfloat xx, GLfloat yy) {
    x = xx;
    y = yy;
}

__device__ __host__
Vector2::Vector2(const Vector2& a) {
    x = a.x;
    y = a.y;
}

__device__ __host__
GLfloat Vector2::magnitudeSquared() {
    return x*x+y*y;
}

__device__ __host__
GLfloat  Vector2::magnitude() {
    return sqrtf(x*x+y*y);
}

__device__ __host__
Vector2 Vector2::normalize() {
    GLfloat invSqrt = 1.0/sqrtf(x*x+y*y);
    return Vector2(x*invSqrt, y*invSqrt);
}

__device__ __host__
Vector2 Vector2::operator+(const Vector2 &b) {
    return Vector2(x+b.x, y+b.y);
}

__device__ __host__
Vector2 Vector2::operator-(const Vector2 &b) {
    return Vector2(x-b.x, y-b.y);
}

__device__ __host__
Vector2 Vector2::operator*(const GLfloat &s) {
    return Vector2(x*s, y*s);
}

std::ostream& operator<<(std::ostream& stream, const Vector2& vector) {
    stream << vector.x << ", " << vector.y;
    return stream;
}