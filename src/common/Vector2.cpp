// Ben Bennett
// CIS 531
// Implementation of the Vector2 class
#include "Vector2.h"
#include <math.h>
#include <iostream>

Vector2::Vector2(GLfloat xx, GLfloat yy) {
    x = xx;
    y = yy;
}

Vector2::Vector2(const Vector2& a) {
    x = a.x;
    y = a.y;
}

GLfloat Vector2::magnitudeSquared() {
    return x*x+y*y;
}

GLfloat  Vector2::magnitude() {
    return sqrtf(x*x+y*y);
}

Vector2 Vector2::normalize() {
    GLfloat invSqrt = 1.0/sqrtf(x*x+y*y);
    return Vector2(x*invSqrt, y*invSqrt);
}

Vector2 Vector2::operator+(const Vector2 &b) {
    return Vector2(x+b.x, y+b.y);
}

Vector2 Vector2::operator-(const Vector2 &b) {
    return Vector2(x-b.x, y-b.y);
}

Vector2 Vector2::operator*(const GLfloat &s) {
    return Vector2(x*s, y*s);
}

// Output operator for easy debugging
std::ostream& operator<<(std::ostream& stream, const Vector2& vector) {
    stream << vector.x << ", " << vector.y;
    return stream;
}