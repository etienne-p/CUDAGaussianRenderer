#include "CameraControls.h"

static bool
RayIntersectsPlane(const glm::vec3& ray_origin, const glm::vec3& ray_direction, const glm::vec4& plane, float& t)
{
    auto denom = glm::dot(ray_direction, plane.xyz());
    if (denom == 0.0f)
    {
        return false;
    }

    t = -(glm::dot(ray_origin, plane.xyz()) + plane.w) / denom;
    return true;
}

static glm::vec4 GetPlane(const glm::vec3& normal, const glm::vec3& point)
{
    return glm::vec4(normal, -glm::dot(normal, point));
}

static glm::vec3 ProjectOnPlane(const glm::vec3& vec, const glm::vec3& plane_normal)
{
    return vec - glm::dot(plane_normal, vec);
}

static constexpr glm::vec3 kUp = glm::vec3(0, 1, 0);
static constexpr glm::vec3 kRight = glm::vec3(1, 0, 0);
static constexpr glm::vec3 kBack = glm::vec3(0, 0, 1);

static glm::quat RemoveRoll(glm::quat rotation)
{
    auto rot = glm::mat3(rotation);
    auto right = rot[0];
    auto up = rot[1];
    auto forward = rot[2];

    right = glm::normalize(ProjectOnPlane(right, kUp));
    up = glm::normalize(ProjectOnPlane(up, right));
    forward = glm::cross(right, up);

    return glm::quat(glm::mat3(right, up, forward));
}

glm::vec3 CameraControls::GetMovement() const
{
    auto result = glm::vec3(0.0f);

    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
    {
        result.z -= 1.0f;
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
    {
        result.x -= 1.0f;
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
    {
        result.z += 1.0f;
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
    {
        result.x += 1.0f;
    }
    if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS)
    {
        result.y += 1.0f;
    }
    if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS)
    {
        result.y -= 1.0f;
    }
    return result;
}

glm::vec3 CameraControls::GetPosition() const
{
    return position_;
}

glm::mat4 CameraControls::GetView() const
{
    auto rotation = glm::mat4_cast(rotation_);
    auto translation = glm::translate(glm::mat4(1.0f), position_);
    return glm::inverse(translation * rotation);
}

glm::mat4 CameraControls::GetProjection() const
{
    return glm::perspective(field_of_view_, GetAspect(), z_near_, z_far_);
}

glm::mat4 CameraControls::GetViewProjection() const
{
    return GetProjection() * GetView();
}

void CameraControls::GetWorldSpaceRay(
    const glm::vec2& pointer_position, glm::vec3& ray_origin, glm::vec3& ray_direction) const
{
    auto view_projection = GetViewProjection();
    auto view_projection_inverse = glm::inverse(view_projection);
    auto pointer_position_clip = (pointer_position / screen_size_) * 2.0f - 1.0f;
    pointer_position_clip.y *= -1;
    auto from = view_projection_inverse * glm::vec4(pointer_position_clip, -1.0f, 1.0f);
    auto to = view_projection_inverse * glm::vec4(pointer_position_clip, 1.0f, 1.0f);
    from /= from.w;
    to /= to.w;

    ray_origin = from;
    ray_direction = glm::normalize(to - from);
}

CameraControls::MouseButton CameraControls::GetMouseButton() const
{
    if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        return CameraControls::MouseButton::Left;
    }
    if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        return CameraControls::MouseButton::Middle;
    }
    if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        return CameraControls::MouseButton::Right;
    }
    return CameraControls::MouseButton::None;
}

void CameraControls::SetBounds(const glm::vec3& min, const glm::vec3& max)
{
    auto size = max - min;
    auto center = min + size * 0.5f;
    auto max_size = glm::max(size.x, glm::max(size.y, size.z));
    speed_ = max_size * 0.02f;
    auto offset = glm::normalize(glm::vec3(0.0f, max_size * 0.5f, max_size * 0.5f));
    auto dist = glm::sqrt(2.0f) * max_size * 0.5f / glm::tan(field_of_view_ * 0.5f);
    position_ = center + offset * dist;
    rotation_ = glm::quatLookAt(-offset, kUp);
    floor_plane_ = GetPlane(kUp, center);
    anchor_position_ = center;
}

void CameraControls::Update(float dt)
{
    auto mouse_button = GetMouseButton();
    if (mouse_button != mouse_button_)
    {
        switch (mouse_button)
        {
            case MouseButton::None:
                mode_ = Mode::None;
                break;
            case MouseButton::Left:
                mode_ = Mode::Drag;
                break;
            case MouseButton::Middle:
                mode_ = Mode::Orbit;
                break;
            case MouseButton::Right:
                mode_ = Mode::Pan;
                start_pan_position_is_valid_ = false;
                break;
        }

        mouse_button_ = mouse_button;
    }

    double xpos, ypos;
    glfwGetCursorPos(window_, &xpos, &ypos);
    auto pointer_is_inside = xpos >= 0 || xpos < screen_size_.x || ypos >= 0 || ypos < screen_size_.y;
    auto pointer_position = glm::vec2((float) xpos, (float) ypos);

    auto ray_origin = glm::vec3();
    auto ray_direction = glm::vec3();
    GetWorldSpaceRay(pointer_position, ray_origin, ray_direction);

    auto pointer_delta = mouse_position_is_valid_ ? pointer_position - pointer_position_ : glm::vec2(0.0f);

    mouse_position_is_valid_ = true;

    switch (mode_)
    {
        case Mode::None: {
            auto t = 0.0f;
            if (RayIntersectsPlane(ray_origin, ray_direction, floor_plane_, t))
            {
                anchor_position_ = ray_origin + ray_direction * t;
            }
        }
        break;

        case Mode::Drag: {
            auto yaw_and_pitch = glm::vec2(field_of_view_ * GetAspect(), field_of_view_) * pointer_delta / screen_size_;
            auto yaw = glm::angleAxis(yaw_and_pitch.x, kUp);
            auto pitch = glm::angleAxis(yaw_and_pitch.y, kRight);
            rotation_ = RemoveRoll(rotation_ * yaw * pitch);
        }
        break;

        case Mode::Orbit: {
            auto yaw_and_pitch = glm::vec2(field_of_view_ * GetAspect(), field_of_view_) * pointer_delta / screen_size_;
            auto right = rotation_ * kRight;
            auto pitch_rot = glm::angleAxis(-yaw_and_pitch.y, right);
            auto yaw_rot = glm::angleAxis(-yaw_and_pitch.x, kUp);
            auto delta_rotation = yaw_rot * pitch_rot;
            auto new_rotation = RemoveRoll(delta_rotation * rotation_);
            auto anchor_to_camera = delta_rotation * (position_ - anchor_position_);
            rotation_ = new_rotation;
            position_ = anchor_position_ + anchor_to_camera;
        }
        break;

        case Mode::Pan: {
            auto plane = GetPlane(rotation_ * kBack, anchor_position_);
            auto t = 0.0f;

            if (RayIntersectsPlane(ray_origin, ray_direction, plane, t))
            {
                auto hit = ray_origin + ray_direction * t;
                if (start_pan_position_is_valid_)
                {
                    auto delta = hit - start_pan_position_;
                    position_ -= delta;
                }
                else if (pointer_is_inside)
                {
                    start_pan_position_ = hit;
                    start_pan_position_is_valid_ = true;
                }
            }
        }
        break;
    }

    position_ += rotation_ * (GetMovement() * dt);

    pointer_position_ = pointer_position;
}
