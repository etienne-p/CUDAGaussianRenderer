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

    GLFWwindow* window_{nullptr};
    Mode mode_{Mode::None};
    bool mouse_position_is_valid_{false};
    MouseButton mouse_button_{MouseButton::None};
    glm::vec2 pointer_position_;
    glm::vec3 start_pan_position_;
    bool start_pan_position_is_valid_{false};
    glm::vec2 screen_size_;

    float z_near_{0.1f};
    float z_far_{100.0f};
    float field_of_view_{glm::radians(60.0f)};

    float speed_;
    glm::vec3 anchor_position_;
    glm::vec4 floor_plane_;
    glm::quat rotation_;
    glm::vec3 position_;

    glm::vec3 GetMovement() const;
    MouseButton GetMouseButton() const;
    void GetWorldSpaceRay(const glm::vec2& pointer_position, glm::vec3& ray_origin, glm::vec3& ray_direction) const;

  public:
    CameraControls(GLFWwindow* window, const glm::vec2& screen_size)
        : window_(window)
        , screen_size_(screen_size)
    {}
    float GetAspect() const
    {
        return screen_size_.x / screen_size_.y;
    }
    float GetFieldOfView() const
    {
        return field_of_view_;
    }
    float GetNear() const
    {
        return z_near_;
    }
    float GetFar() const
    {
        return z_far_;
    }
    glm::vec3 GetPosition() const;
    glm::mat4 GetView() const;
    glm::mat4 GetProjection() const;
    glm::mat4 GetViewProjection() const;
    void SetBounds(const glm::vec3& min, const glm::vec3& max);
    void Update(float dt);
};
