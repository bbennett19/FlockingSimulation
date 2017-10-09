// Ben Bennett
// CIS 531
// Class definition for WindowManager
#ifndef WINDOWMANAGER_H
#define WINDOWMANAGER_H

// A window management class.
class WindowManager {
private:
    Display *display;
    Window win;
    GLXContext ctx;
    Colormap cmap;
public:
    void createWindow();
    void shutdown();
    bool quit();
    Display* getDisplay();
    Window getWindow();
};

#endif
