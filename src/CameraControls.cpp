#include "CameraControls.h"

bool rayIntersectsPlane(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, const glm::vec4& plane, float& t)
{
    auto denom = glm::dot(rayDirection, plane.xyz());
    if (denom == 0.0f)
    {
        return false;
    }

    t = -(glm::dot(rayOrigin, plane.xyz()) + plane.w) / denom;
    return true;
}

glm::vec4 getPlane(const glm::vec3& normal, const glm::vec3& point)
{
    return glm::vec4(normal, -glm::dot(normal, point));
}

glm::vec3 projectOnPlane(const glm::vec3& vec, const glm::vec3& planeNormal)
{
    return vec - glm::dot(planeNormal, vec);
}

constexpr glm::vec3 k_Up = glm::vec3(0, 1, 0);
constexpr glm::vec3 k_Right = glm::vec3(1, 0, 0);
constexpr glm::vec3 k_Back = glm::vec3(0, 0, 1);

glm::quat removeRoll(glm::quat rotation)
{
    auto rot = glm::mat3(rotation);
    auto right = rot[0];
    auto up = rot[1];
    auto forward = rot[2];

    right = glm::normalize(projectOnPlane(right, k_Up));
    up = glm::normalize(projectOnPlane(up, right));
    forward = glm::cross(right, up);

    return glm::quat(glm::mat3(right, up, forward));
}

glm::vec3 CameraControls::getMovement() const
{
    auto result = glm::vec3(0.0f);

    if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS)
    {
        result.z -= 1.0f;
    }
    if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS)
    {
        result.x -= 1.0f;
    }
    if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS)
    {
        result.z += 1.0f;
    }
    if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS)
    {
        result.x += 1.0f;
    }
    if (glfwGetKey(m_Window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        result.y += 1.0f;
    }
    if (glfwGetKey(m_Window, GLFW_KEY_E) == GLFW_PRESS)
    {
        result.y -= 1.0f;
    }
    return result;
}

glm::mat4 CameraControls::getView() const
{
    // Or transpose rotation, subtract translation.
    auto rotation = glm::mat4_cast(m_Rotation);
    auto translation = glm::translate(glm::mat4(1.0f), m_Position);
    auto scale = glm::scale(glm::mat4(1.0f), glm::vec3(1, 1, -1));
    return glm::inverse(translation * rotation); // *scale;
}

glm::mat4 CameraControls::getProjection() const
{
    return glm::perspective(m_FieldOfView, getAspect(), m_Near, m_Far);
}

glm::mat4 CameraControls::getViewProjection() const
{
    return getProjection() * getView();
}

void CameraControls::getWorldSpaceRay(
    const glm::vec2& pointerPosition, glm::vec3& rayOrigin, glm::vec3& rayDirection) const
{
    auto viewProjection = getViewProjection();
    auto viewProjectionInverse = glm::inverse(viewProjection);
    auto pointerPositionClip = (pointerPosition / m_ScreenSize) * 2.0f - 1.0f;
    // Flip y, mouse coords start at top.
    pointerPositionClip.y *= -1;
    auto from = viewProjectionInverse * glm::vec4(pointerPositionClip, -1.0f, 1.0f);
    auto to = viewProjectionInverse * glm::vec4(pointerPositionClip, 1.0f, 1.0f);
    from /= from.w;
    to /= to.w;

    rayOrigin = from;
    rayDirection = glm::normalize(to - from);
}

CameraControls::MouseButton CameraControls::getMouseButton() const
{
    if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        return CameraControls::MouseButton::Left;
    }
    if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        return CameraControls::MouseButton::Middle;
    }
    if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        return CameraControls::MouseButton::Right;
    }
    return CameraControls::MouseButton::None;
}

void CameraControls::setBounds(const glm::vec3& min, const glm::vec3& max)
{
    auto size = max - min;
    auto center = min + size * 0.5f;
    auto maxSize = glm::max(size.x, glm::max(size.y, size.z));
    // Infer speed from bounds.
    m_Speed = maxSize * 0.01f;
    auto offset = glm::vec3(0.0f, maxSize * 0.5f, maxSize * 0.5f);
    m_Position = center + offset;
    m_Rotation = glm::angleAxis(0.0f, k_Right);
    m_FloorPlane = getPlane(k_Up, center);
    m_AnchorPosition = center;
}

void CameraControls::update(float dt)
{
    // Update mode.
    auto mouseButton = getMouseButton();
    if (mouseButton != m_MouseButton)
    {
        switch (mouseButton)
        {
            case MouseButton::None:
                m_Mode = Mode::None;
                break;
            case MouseButton::Left:
                m_Mode = Mode::Drag;
                break;
            case MouseButton::Middle:
                m_Mode = Mode::Orbit;
                break;
            case MouseButton::Right:
                m_Mode = Mode::Pan;
                m_StartPanPositionIsValid = false;
                break;
        }

        m_MouseButton = mouseButton;
    }

    // Fetch pointer position and evaluate world space ray.
    double xpos, ypos;
    glfwGetCursorPos(m_Window, &xpos, &ypos);
    auto pointerIsInside = xpos >= 0 || xpos < m_ScreenSize.x || ypos >= 0 || ypos < m_ScreenSize.y;
    auto pointerPosition = glm::vec2((float) xpos, (float) ypos);

    auto rayOrigin = glm::vec3();
    auto rayDirection = glm::vec3();
    getWorldSpaceRay(pointerPosition, rayOrigin, rayDirection);

    auto pointerDelta = m_MousePositionIsValid ? pointerPosition - m_PointerPosition : glm::vec2(0.0f);

    // Only useful for the first update.
    m_MousePositionIsValid = true;

    // TODO: Add orbit more once we support raycasting.
    switch (m_Mode)
    {
        case Mode::None: {
            auto t = 0.0f;
            if (rayIntersectsPlane(rayOrigin, rayDirection, m_FloorPlane, t))
            {
                m_AnchorPosition = rayOrigin + rayDirection * t;
            }
        }

        break;

        case Mode::Drag: {
            auto yawAndPitch = glm::vec2(m_FieldOfView * getAspect(), m_FieldOfView) * pointerDelta / m_ScreenSize;
            auto yaw = glm::angleAxis(yawAndPitch.x, k_Up);
            auto pitch = glm::angleAxis(yawAndPitch.y, k_Right);
            m_Rotation = removeRoll(m_Rotation * yaw * pitch);
        }

        break;

        case Mode::Orbit: {
            // Pitch and yaw directly derived from pointer movement.
            auto yawAndPitch = glm::vec2(m_FieldOfView * getAspect(), m_FieldOfView) * pointerDelta / m_ScreenSize;
            auto right = m_Rotation * k_Right;
            auto pitchRot = glm::angleAxis(-yawAndPitch.y, right);
            auto yawRot = glm::angleAxis(-yawAndPitch.x, k_Up);
            auto deltaRotation = yawRot * pitchRot;
            auto rotation = removeRoll(deltaRotation * m_Rotation);
            // Move with respect to the anchor.
            auto anchorToCamera = deltaRotation * (m_Position - m_AnchorPosition);
            m_Rotation = rotation;
            m_Position = m_AnchorPosition + anchorToCamera;
        }
        break;

        case Mode::Pan: {
            auto plane = getPlane(m_Rotation * k_Back, m_AnchorPosition);
            auto t = 0.0f;

            if (rayIntersectsPlane(rayOrigin, rayDirection, plane, t))
            {
                auto hit = rayOrigin + rayDirection * t;
                if (m_StartPanPositionIsValid)
                {
                    auto delta = hit - m_StartPanPosition;
                    m_Position -= delta;
                }
                else if (pointerIsInside)
                {
                    m_StartPanPosition = hit;
                    m_StartPanPositionIsValid = true;
                }
            }
        }

        break;
    }

    // Apply movement (W, A, S, D, ...).
    m_Position += m_Rotation * (getMovement() * dt);

    m_PointerPosition = pointerPosition;
}
