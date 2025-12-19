#pragma once

#define GLM_FORCE_SWIZZLE
#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>

class CameraControls
{
  private:
    enum class Mode
    {
        None,
        Pan,
        Orbit,
        Drag
    };

    enum class MouseButton
    {
        None,
        Left,
        Middle,
        Right
    };

    GLFWwindow* m_Window{nullptr};
    Mode m_Mode{Mode::None};
    bool m_MousePositionIsValid{false};
    MouseButton m_MouseButton{MouseButton::None};
    glm::vec2 m_PointerPosition;
    glm::vec3 m_StartPanPosition;
    bool m_StartPanPositionIsValid{false};
    glm::vec2 m_ScreenSize;

    float m_Near{0.1f};
    float m_Far{100.0f};
    float m_FieldOfView{glm::radians(60.0f)};

    float m_Speed;
    glm::vec3 m_AnchorPosition;
    glm::vec4 m_FloorPlane;
    glm::quat m_Rotation;
    glm::vec3 m_Position;

    glm::vec3 getMovement() const;
    MouseButton getMouseButton() const;
    void getWorldSpaceRay(const glm::vec2& pointerPosition, glm::vec3& rayOrigin, glm::vec3& rayDirection) const;

  public:
    CameraControls(GLFWwindow* window, const glm::vec2& screenSize)
        : m_Window(window)
        , m_ScreenSize(screenSize)
    {}
    float getAspect() const
    {
        return m_ScreenSize.x / m_ScreenSize.y;
    }
    float getFieldOfView() const
    {
        return m_FieldOfView;
    }
    float getNear() const
    {
        return m_Near;
    }
    float getFar() const
    {
        return m_Far;
    }
    glm::vec3 getPosition() const;
    glm::mat4 getView() const;
    glm::mat4 getProjection() const;
    glm::mat4 getViewProjection() const;
    void setBounds(const glm::vec3& min, const glm::vec3& max);
    void update(float dt);
};
