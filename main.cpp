// Skeletal Animation example from
// https://learnopengl.com/Guest-Articles/2020/Skeletal-Animation.
// Refactored and simplified.
// 
#include <assimp/Importer.hpp>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/Quaternion.h>
#include <assimp/scene.h>
#include <assimp/vector3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <map>
#include <stack>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <iterator>
#include <algorithm>
#include <filesystem>

#include <cmath>

#if defined(NDEBUG)
#  undef NDEBUG
#endif
#include <cassert>

///////////////////////////////////////////////////////////////////////////////
// ANIMATIONS.

using BoneIndex = int;

struct AnimVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texture_uv;
    BoneIndex bone_ids[4];
    float weights[4];
};

struct KeyPosition
{
    glm::vec3 position;
    float time_stamp;
};

struct KeyRotation
{
    glm::quat orientation;
    float time_stamp;
};

struct KeyScale
{
    glm::vec3 scale;
    float time_stamp;
};

struct BoneKeyFrames
{
    // Constant run-time data.
    BoneIndex _bone_index = -1;
    glm::mat4 _bone_to_model = glm::mat4(1.f);
    std::vector<KeyPosition> _positions;
    std::vector<KeyRotation> _rotations;
    std::vector<KeyScale> _scales;

    // "Optimization". Remember previous frame's state
    // to find next frame keys "faster". Can be removed.
    float _prev_animation_time = -1;
    int _prev_position_index = -1;
    int _prev_rotation_index = -1;
    int _prev_scale_index = -1;

    glm::mat4 update(float animation_time)
    {
        const glm::mat4 translation = interpolate_position(animation_time);
        const glm::mat4 rotation = interpolate_rotation(animation_time);
        const glm::mat4 scale = interpolate_scaling(animation_time);
        _prev_animation_time = animation_time;
        return (translation * rotation * scale);
    }

private:
#if (0)
    template<typename Key>
    static int GetFrameIndex_Simple(const std::vector<Key>& frames, float animation_time)
    {
        assert(frames.size() >= 2);
        for (std::size_t index = 0, count = frames.size(); index < (count - 1); ++index)
        {
            if (animation_time < frames[index + 1].time_stamp)
            {
                return int(index);
            }
        }
        assert(false);
        return -1;
    }
#else
    template<typename Key>
    struct KeyTimeCompare
    {
        bool operator()(const Key& lhs, const Key& rhs) const noexcept
        {
            return (lhs.time_stamp < rhs.time_stamp);
        }
        bool operator()(float animation_time, const Key& rhs) const noexcept
        {
            return (animation_time < rhs.time_stamp);
        }
        bool operator()(const Key& lhs, float animation_time) const noexcept
        {
            return (lhs.time_stamp < animation_time);
        }
    };

    // Same as simplified version above, but does
    // (a) binary search and (b) remembers last frame's
    // results to go thru smaller range of values.
    template<typename Key>
    static int GetFrameIndex(const std::vector<Key>& frames
        , float animation_time, unsigned start_offset, unsigned end_offset)
    {
        assert(frames.size() >= 2);
        auto it = std::lower_bound(
              frames.cbegin() + start_offset
            , frames.cbegin() + end_offset
            , animation_time
            , KeyTimeCompare<Key>{});
        // Below, even tho we searched in [start_offset, end_offset) range
        // the index should be in [0, frames.size()), hence the use of
        // begin()/end() iterators pair, discarding search range.
        if (it == frames.cbegin()) // most-likely, zero.
        {
            it = frames.cbegin() + 1;
        }
        // If fired, means that `animationTime` is outside
        // of [0, full_animation_time] range. Precondition requirement violation.
        assert(it != frames.cend());
        const int index = (int(std::distance(frames.cbegin(), it)) - 1);
        assert(index >= 0);
        assert(index < (int(frames.size()) - 1));
        assert(frames[index].time_stamp <= animation_time);
        assert(frames[index + 1].time_stamp >= animation_time);
        return index;
    }
#endif

    // Cached version of GetFrameIndex().
    template<typename Key>
    static int UpdateFrameIndex(const std::vector<Key>& frames
        , float animation_time, int prev_known_index, float prev_animation_time)
    {
#if (0)
        (void)prev_animation_time;
        (void)prev_known_index;
        return GetFrameIndex_Simple(frames, animation_time);
#else
        assert(prev_known_index < int(frames.size()));
        if (prev_known_index < 0)
        {
            // First time running, examine full range [0, size).
            return GetFrameIndex(frames, animation_time, 0, unsigned(frames.size()));
        }
        assert(prev_known_index >= 0);
        assert(prev_animation_time >= 0);
        if (animation_time >= prev_animation_time)
        {
            return GetFrameIndex(frames, animation_time, prev_known_index, unsigned(frames.size()));
        }
        // else: animation_time < prev_animation_time
        return GetFrameIndex(frames, animation_time, 0, prev_known_index);
#endif
    }

    static float GetScaleFactor(float prev_time_stamp, float next_time_stamp, float animation_time)
    {
        assert(animation_time >= prev_time_stamp);
        assert(next_time_stamp > prev_time_stamp);
        const float progress = animation_time - prev_time_stamp;
        const float total = next_time_stamp - prev_time_stamp;
        assert(progress <= total);
        return progress / total;
    }

    glm::mat4 interpolate_position(float animation_time)
    {
        const int p0 = UpdateFrameIndex(_positions, animation_time, _prev_position_index, _prev_animation_time);
        _prev_position_index = p0;
        const KeyPosition& prev = _positions[p0];
        const KeyPosition& next = _positions[p0 + 1];
        const float scale_factor = GetScaleFactor(prev.time_stamp, next.time_stamp, animation_time);
        const glm::vec3 position = glm::mix(prev.position, next.position, scale_factor);
        return glm::translate(glm::mat4(1.0f), position);
    }

    glm::mat4 interpolate_rotation(float animation_time)
    {
        const int p0 = UpdateFrameIndex(_rotations, animation_time, _prev_rotation_index, _prev_animation_time);
        _prev_rotation_index = p0;
        const KeyRotation& prev = _rotations[p0];
        const KeyRotation& next = _rotations[p0 + 1];
        const float scale_factor = GetScaleFactor(prev.time_stamp, next.time_stamp, animation_time);
        const glm::quat rotation = glm::normalize(glm::slerp(
            prev.orientation, next.orientation, scale_factor));
        return glm::toMat4(rotation);
    }

    glm::mat4 interpolate_scaling(float animation_time)
    {
        const int p0 = UpdateFrameIndex(_scales, animation_time, _prev_scale_index, _prev_animation_time);
        _prev_scale_index = p0;
        const KeyScale& prev = _scales[p0];
        const KeyScale& next = _scales[p0 + 1];
        const float scale_factor = GetScaleFactor(prev.time_stamp, next.time_stamp, animation_time);
        const glm::vec3 scale = glm::mix(prev.scale, next.scale, scale_factor);
        return glm::scale(glm::mat4(1.0f), scale);
    }
};

struct AnimNode
{
    // Runtime. Updated every frame.
    std::optional<BoneKeyFrames> bone;
    glm::mat4 frame_transform;

    // Constant data.
    int parent = -1;
    glm::mat4 bone_transform;
};

class Animation
{
public:
    // Limit from Vertex shader.
    static constexpr std::size_t kMaxBonesCount = 100;

    explicit Animation(std::vector<AnimNode>&& nodes, float duration, float ticks_per_second)
        : _transforms(kMaxBonesCount, glm::mat4(1.0f))
        , _nodes(std::move(nodes))
        , _current_time(0.f)
        , _duration(duration)
        , _ticks_per_second(ticks_per_second)
    {
    }

    void update(float dt)
    {
        _current_time += _ticks_per_second * dt;
        _current_time = fmod(_current_time, _duration);

        for (std::size_t i = 0, count = _nodes.size(); i < count; ++i)
        {
            AnimNode& node = _nodes[i];
            assert(int(i) > node.parent);

            const glm::mat4 node_transform = node.bone
                ? node.bone->update(_current_time)
                : node.bone_transform;
            const glm::mat4 parent_transform = (node.parent >= 0)
                ? _nodes[node.parent].frame_transform
                : glm::mat4(1.0f);
            const glm::mat4 global_transformation = parent_transform * node_transform;
            node.frame_transform = global_transformation;

            if (node.bone)
            {
                const std::size_t bone_index = node.bone->_bone_index;
                assert(bone_index < _transforms.size()
                    && "Too many bones. See kMaxBonesCount limit.");
                _transforms[bone_index] = global_transformation * node.bone->_bone_to_model;
            }
        }
    }

    const std::vector<glm::mat4>& transforms() const
    { 
        return _transforms;
    }

private:
    std::vector<glm::mat4> _transforms;
    std::vector<AnimNode> _nodes;
    float _current_time;
    float _duration;
    float _ticks_per_second;
};

///////////////////////////////////////////////////////////////////////////////
// ASSIMP.
#if defined(ASSIMP_DOUBLE_PRECISION)
#  error ASSIMP vs GLM: Need to convert double to float.
#endif

static glm::mat4 Matrix_RowToColumn(const aiMatrix4x4& m)
{
    const glm::vec4 c1(m.a1, m.b1, m.c1, m.d1);
    const glm::vec4 c2(m.a2, m.b2, m.c2, m.d2);
    const glm::vec4 c3(m.a3, m.b3, m.c3, m.d3);
    const glm::vec4 c4(m.a4, m.b4, m.c4, m.d4);
    return glm::mat4(c1, c2, c3, c4);
}

static glm::vec3 Vec_ToGLM(const aiVector3D& vec)
{
    return glm::vec3(vec.x, vec.y, vec.z);
}

static glm::quat Quat_ToGLM(const aiQuaternion& pOrientation)
{
    return glm::quat(pOrientation.w, pOrientation.x, pOrientation.y, pOrientation.z);
}

enum class TextureType
{
    Invalid, Diffuse, Specular, Normal, Height
};

struct AnimTexture
{
    std::filesystem::path file_path;
    TextureType type = TextureType::Diffuse;
};

struct AnimMesh
{
    std::vector<AnimVertex> vertices;
    std::vector<unsigned> indices;
    std::vector<AnimTexture> textures;
};

struct BoneMeshInfo
{
    // Unique bone index used in vertex shader to index transforms.
    BoneIndex index = -1;
    // Inverse-bind matrix or inverse bind pose matrix or "offset" matrix:
    // https://stackoverflow.com/questions/50143649/what-does-moffsetmatrix-actually-do-in-assimp.
    glm::mat4 bone_to_model;
};

// Helper to remap bones with string names to indexes to array.
// Used while loading ASSIMP model. Not needed after loading.
struct BoneInfoRemap
{
    std::map<std::string, BoneMeshInfo, std::less<>> _name_to_info;
    BoneIndex _next_bone_id = 0;

    BoneIndex add_new_bone(std::string&& name, glm::mat4 bone_to_model)
    {
        auto [it, inserted] = _name_to_info.insert(
            std::make_pair(std::move(name), BoneMeshInfo{-1, bone_to_model}));
        assert(inserted && "Inserting duplicated bone.");
        BoneMeshInfo& info = it->second;
        info.index = _next_bone_id++;
        return info.index;
    }

    const BoneMeshInfo* get(const char* name) const
    {
        auto it = _name_to_info.find(name);
        return ((it != _name_to_info.end()) ? &(it->second) : nullptr);
    }
};

static AnimMesh Assimp_LoadMesh(
    const std::filesystem::path& model_path
    , const aiScene& scene
    , const aiMesh& mesh
    , BoneInfoRemap& bone_info)
{
    // Vertices.
    std::vector<AnimVertex> vertices;
    vertices.reserve(mesh.mNumVertices);
    for (unsigned i = 0; i < mesh.mNumVertices; ++i)
    {
        aiVector3D* uvs = mesh.mTextureCoords[0];
        assert(uvs);
        assert(mesh.mNormals);
        vertices.push_back(AnimVertex{});
        AnimVertex& v = vertices.back();
        v.position = Vec_ToGLM(mesh.mVertices[i]);
        v.normal = Vec_ToGLM(mesh.mNormals[i]);
        v.texture_uv = glm::vec2(uvs[i].x, uvs[i].y);
        std::fill(std::begin(v.bone_ids), std::end(v.bone_ids), -1);
        std::fill(std::begin(v.weights), std::end(v.weights), 0.f);
    }
    // Indices.
    std::vector<unsigned> indices;
    indices.reserve(mesh.mNumFaces * std::size_t(3));
    for (unsigned i = 0; i < mesh.mNumFaces; ++i)
    {
        aiFace face = mesh.mFaces[i];
        assert(face.mNumIndices == 3);
        for (unsigned j = 0; j < face.mNumIndices; ++j)
        {
            indices.push_back(face.mIndices[j]);
        }
    }
    // Textures.
    const aiMaterial* const material = scene.mMaterials[mesh.mMaterialIndex];
    assert(material);

    const struct TextureInfo
    {
        aiTextureType assimp_type;
        TextureType type;
    } kTexturesToFind[] =
    {
        {aiTextureType_DIFFUSE, TextureType::Diffuse},
        {aiTextureType_SPECULAR, TextureType::Specular},
        {aiTextureType_HEIGHT, TextureType::Normal},
        {aiTextureType_AMBIENT, TextureType::Height},
    };
    std::vector<AnimTexture> textures;
    for (auto [assimp_type, type] : kTexturesToFind)
    {
        for (unsigned i = 0, count = material->GetTextureCount(assimp_type); i < count; ++i)
        {
            aiString file_name;
            if (material->GetTexture(assimp_type, i, &file_name) == aiReturn_SUCCESS)
            {
                textures.push_back({});
                AnimTexture& t = textures.back();
                t.file_path = model_path.parent_path() / std::string(file_name.data, file_name.length);
                t.type = type;
                assert(std::filesystem::exists(t.file_path));
            }
        }
    }

    // Bones weights for each vertex.
    auto add_bone_weight_to_vertex = [](AnimVertex& vertex, BoneIndex bone_index, float weight)
    {
        // Up to 4 influences supported. See AnimVertex passed to Vertex shader.
        auto it = std::find_if(std::begin(vertex.bone_ids), std::end(vertex.bone_ids)
            , [&](BoneIndex index) { return (index < 0) && (index != bone_index); });
        assert(it != std::end(vertex.bone_ids)
            && "Either more then 4 bones per vertex OR duplicated bone.");
        const std::size_t i = std::distance(std::begin(vertex.bone_ids), it);
        vertex.weights[i] = weight;
        vertex.bone_ids[i] = bone_index;
    };

    for (unsigned i = 0; i < mesh.mNumBones; ++i)
    {
        const aiString& bone_name = mesh.mBones[i]->mName;
        const BoneIndex bone_index = bone_info.add_new_bone(
            std::string(bone_name.data, bone_name.length)
            , Matrix_RowToColumn(mesh.mBones[i]->mOffsetMatrix));
        const aiBone* const bone = mesh.mBones[i];
        assert(bone);
        const aiVertexWeight* const weights = bone->mWeights;
        for (unsigned j = 0; j < bone->mNumWeights; ++j)
        {
            const unsigned vertex_id = weights[j].mVertexId;
            const float weight = weights[j].mWeight;
            assert(vertex_id <= vertices.size());
            add_bone_weight_to_vertex(vertices[vertex_id], bone_index, weight);
        }
    }

    AnimMesh anim_mesh;
    anim_mesh.indices = std::move(indices);
    anim_mesh.vertices = std::move(vertices);
    anim_mesh.textures = std::move(textures);
    return anim_mesh;
}

static std::vector<AnimMesh> Assimp_LoadModelMeshWithAnimationsWeights(
    const std::filesystem::path& model_path
    , const aiScene& scene
    , BoneInfoRemap& bone_info)
{
    std::vector<AnimMesh> meshes;

    std::stack<const aiNode*> dfs;
    dfs.push(scene.mRootNode);
    while (dfs.size() > 0)
    {
        const aiNode* const node = dfs.top();
        dfs.pop();

        for (unsigned i = 0; i < node->mNumMeshes; ++i)
        {
            const aiMesh* const mesh = scene.mMeshes[node->mMeshes[i]];
            assert(mesh);
            meshes.push_back(Assimp_LoadMesh(model_path, scene, *mesh, bone_info));
        }
        for (unsigned i = 0; i < node->mNumChildren; ++i)
        {
            dfs.push(node->mChildren[i]);
        }
    }
    return meshes;
}

static BoneKeyFrames Assimp_LoadBoneKeyFrames(const aiNodeAnim& channel, const BoneMeshInfo& bone_info)
{
    BoneKeyFrames bone;
    bone._bone_index = bone_info.index;
    bone._bone_to_model = bone_info.bone_to_model;

    bone._positions.reserve(channel.mNumPositionKeys);
    for (unsigned index = 0; index < channel.mNumPositionKeys; ++index)
    {
        KeyPosition data;
        data.position = Vec_ToGLM(channel.mPositionKeys[index].mValue);
        data.time_stamp = float(channel.mPositionKeys[index].mTime);
        bone._positions.push_back(data);
    }

    bone._rotations.reserve(channel.mNumRotationKeys);
    for (unsigned index = 0; index < channel.mNumRotationKeys; ++index)
    {
        KeyRotation data;
        data.orientation = Quat_ToGLM(channel.mRotationKeys[index].mValue);
        data.time_stamp = float(channel.mRotationKeys[index].mTime);
        bone._rotations.push_back(data);
    }

    bone._scales.reserve(channel.mNumScalingKeys);
    for (unsigned index = 0; index < channel.mNumScalingKeys; ++index)
    {
        KeyScale data;
        data.scale = Vec_ToGLM(channel.mScalingKeys[index].mValue);
        data.time_stamp = float(channel.mScalingKeys[index].mTime);
        bone._scales.push_back(data);
    }
    
    return bone;
}

static Animation Assimp_LoadAnimation(const aiScene& scene, const BoneInfoRemap& bone_info)
{
    assert(scene.mNumAnimations == 1);
    const aiAnimation* const animation = scene.mAnimations[0];
    const float duration = float(animation->mDuration);
    const float ticks_per_second = float(animation->mTicksPerSecond);
    std::vector<AnimNode> nodes;
    std::vector<const aiString*> node_names;

    struct Node
    {
        const aiNode* src = nullptr;
        int parent = -1;
    };

    std::stack<Node> dfs;
    dfs.push(Node{scene.mRootNode, -1/*no parent*/});
    while (dfs.size() > 0)
    {
        Node data = std::move(dfs.top());
        dfs.pop();

        AnimNode node;
        node.parent = data.parent;
        node.bone_transform = Matrix_RowToColumn(data.src->mTransformation);
        assert(node.parent < int(nodes.size()));
        nodes.push_back(node);
        node_names.push_back(&data.src->mName);
        const int parent_index = int(nodes.size() - 1);

        for (unsigned i = 0; i < data.src->mNumChildren; ++i)
        {
            dfs.push(Node{data.src->mChildren[i], parent_index});
        }
    }

    for (unsigned i = 0; i < animation->mNumChannels; ++i)
    {
        auto channel = animation->mChannels[i];
        const aiString& bone_name = channel->mNodeName;
        auto it = std::find_if(node_names.cbegin(), node_names.cend()
            , [&bone_name](const aiString* node_name)
        {
            return (bone_name == *node_name);
        });
        assert(it != node_names.end() && "No node matching a bone.");
        const int index = int(std::distance(node_names.cbegin(), it));
        const BoneMeshInfo* _bone_info = bone_info.get(bone_name.C_Str());
        assert(_bone_info && "No bone info remap matching a bone.");

        AnimNode& node = nodes[index];
        assert(not node.bone.has_value() && "Two or more bones matching same node.");
        node.bone.emplace(Assimp_LoadBoneKeyFrames(*channel, *_bone_info));
    }

    return Animation(std::move(nodes), duration, ticks_per_second);
}

///////////////////////////////////////////////////////////////////////////////
// RENDER.
struct RenderTexture
{
    unsigned texture_name;
    TextureType type;

    static RenderTexture FromMemory(TextureType type, GLenum format, int width, int height, const void* data)
    {
        unsigned texture_name = 0;
        glGenTextures(1, &texture_name);
        glBindTexture(GL_TEXTURE_2D, texture_name);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        return RenderTexture(texture_name, type);
    }
    ~RenderTexture() noexcept
    {
        if (std::exchange(type, TextureType::Invalid) != TextureType::Invalid)
        {
            glDeleteTextures(1, &texture_name);
            texture_name = 0;
        }
    }
    RenderTexture(RenderTexture&& rhs) noexcept
        : texture_name(std::exchange(rhs.texture_name, 0))
        , type(std::exchange(rhs.type, TextureType::Invalid)) { }
    RenderTexture& operator=(RenderTexture&&) = delete;
    RenderTexture(const RenderTexture&) = delete;
    RenderTexture& operator=(const RenderTexture&) = delete;
private:
    explicit RenderTexture(unsigned name, TextureType type)
        : texture_name(name), type(type)
    {
        assert(type != TextureType::Invalid);
    }
};

struct TextureLocation
{
    int texture_unit = -1;
    int location = -1;
};

class RenderMesh
{
public:
    static RenderMesh FromMemory(std::vector<AnimVertex>&& vertices
        , std::vector<unsigned>&& indices
        , std::vector<RenderTexture>&& textures)
    {
        assert(vertices.size() > 0);
        assert(indices.size() > 0);
        unsigned VAO = 0;
        unsigned VBO = 0;
        unsigned EBO = 0;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER
            , vertices.size() * sizeof(AnimVertex)
            , vertices.data()
            , GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER
            , indices.size() * sizeof(unsigned)
            , indices.data()
            , GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3/*vec3*/, GL_FLOAT, GL_FALSE
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, position));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3/*vec3*/, GL_FLOAT, GL_FALSE
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2/*vec2*/, GL_FLOAT, GL_FALSE
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, texture_uv));
        glEnableVertexAttribArray(3);
        glVertexAttribIPointer(3, 4/*int[4]*/, GL_INT
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, bone_ids));
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4/*float[4]*/, GL_FLOAT, GL_FALSE
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, weights));
        glBindVertexArray(0);

        return RenderMesh(VAO, VBO, EBO
            , indices.size(), GetTexture(textures, TextureType::Diffuse));
    }

    RenderMesh(const RenderMesh&) = delete;
    RenderMesh& operator=(const RenderMesh&) = delete;
    RenderMesh& operator=(RenderMesh&&) = delete;
    RenderMesh(RenderMesh&& rhs) noexcept
        : _VAO(std::exchange(rhs._VAO, 0))
        , _VBO(std::exchange(rhs._VBO, 0))
        , _EBO(std::exchange(rhs._EBO, 0))
        , _indicies_count(std::exchange(rhs._indicies_count, 0))
        , _diffuse(std::move(rhs._diffuse))
    {
    }
    ~RenderMesh()
    {
        if (std::exchange(_VAO, 0) != 0)
        {
            glDeleteVertexArrays(1, &_VAO);
            glDeleteBuffers(1, &_VBO);
            glDeleteBuffers(1, &_EBO);
        }
    }

    static RenderTexture GetTexture(std::vector<RenderTexture>& textures, TextureType type)
    {
        auto it = std::find_if(textures.begin(), textures.end()
            , [&](const RenderTexture& t)
        {
            return t.type == type;
        });
        assert(it != textures.end() && "No diffuse texture found.");
        return std::move(*it);
    }

    void draw(TextureLocation diffuse)
    {
        glBindVertexArray(_VAO);
        glActiveTexture(GL_TEXTURE0 + diffuse.texture_unit);
        glUniform1i(diffuse.location, _diffuse.texture_name);
        glBindTexture(GL_TEXTURE_2D, _diffuse.texture_name);
        glDrawElements(GL_TRIANGLES, GLsizei(_indicies_count), GL_UNSIGNED_INT, 0);
    }

private:
    explicit RenderMesh(unsigned VAO, unsigned VBO, unsigned EBO
        , std::size_t indicies_count, RenderTexture&& diffuse)
        : _VAO(VAO)
        , _VBO(VBO)
        , _EBO(EBO)
        , _indicies_count(indicies_count)
        , _diffuse(std::move(diffuse))
    {
    }
    unsigned _VAO;
    unsigned _VBO;
    unsigned _EBO;
    std::size_t _indicies_count;
    RenderTexture _diffuse;
};

static RenderTexture OpenGL_LoadTexture(const AnimTexture& raw_texture)
{
    int width = 0;
    int height = 0;
    int components = 0;
    unsigned char* data = stbi_load(raw_texture.file_path.string().c_str()
        , &width, &height, &components, 0);
    assert(data);
    GLenum format = GL_RED;
    switch (components)
    {
    case 1: format = GL_RED; break;
    case 3: format = GL_RGB; break;
    case 4: format = GL_RGBA; break;
    default: assert(false); break;
    }
    auto texture = RenderTexture::FromMemory(raw_texture.type, format, width, height, data);
    stbi_image_free(data);
    return texture;
}

struct OpenGL_ShaderProgram
{
    unsigned _id = 0;

    static OpenGL_ShaderProgram FromBuffers(std::string_view vs, std::string_view ps)
    {
        unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
        assert(vertex != 0);
        const GLint vs_length = GLint(vs.size());
        const char* const vs_ptr = vs.data();
        glShaderSource(vertex, 1, &vs_ptr, &vs_length);
        glCompileShader(vertex);
        {
            int success = -1;
            glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
            assert(success != 0);
        }
        unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
        assert(fragment != 0);
        const GLint ps_length = GLint(ps.size());
        const char* const ps_ptr = ps.data();
        glShaderSource(fragment, 1, &ps_ptr, &ps_length);
        glCompileShader(fragment);
        {
            int success = -1;
            glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
            assert(success != 0);
        }

        unsigned int ID = glCreateProgram();
        assert(ID != 0);
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        {
            int success = -1;
            glGetProgramiv(ID, GL_LINK_STATUS, &success);
            assert(success != 0);
        }
        glDeleteShader(vertex);
        glDeleteShader(fragment);

        return OpenGL_ShaderProgram(ID);
    }

    ~OpenGL_ShaderProgram() noexcept
    {
        if (std::exchange(_id, 0) != 0)
        {
            glDeleteProgram(_id);
        }
    }
    OpenGL_ShaderProgram(OpenGL_ShaderProgram&& rhs) noexcept
        : _id(std::exchange(rhs._id, 0))
    {
    }
    OpenGL_ShaderProgram& operator=(OpenGL_ShaderProgram&&) = delete;
    OpenGL_ShaderProgram(const OpenGL_ShaderProgram&) = delete;
    OpenGL_ShaderProgram& operator=(const OpenGL_ShaderProgram&) = delete;
private:
    explicit OpenGL_ShaderProgram(unsigned id)
        : _id(id)
    {
    }
};

struct RenderModel
{
    static RenderModel Make_NoLighting(std::vector<RenderMesh>&& meshes)
    {
        const char* vertex_shader = R"(
#version 330 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 norm;
layout(location = 2) in vec2 uv;
layout(location = 3) in ivec4 bone_ids;
layout(location = 4) in vec4 weights;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

// See kMaxBonesCount.
uniform mat4 bone_transforms[100];

out vec2 TexCoords;

void main()
{
    vec4 position = vec4(0.0f);
    for (int i = 0; i < 4 ; ++i)
    {
        if (bone_ids[i] == -1)
        {
            continue;
        }
        vec4 local = bone_transforms[bone_ids[i]] * vec4(pos,1.0f);
        position += local * weights[i];
    }

    gl_Position =  projection * view * model * position;
    TexCoords = uv;
}
)";
        const char* fragment_shader = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D diffuse;

void main()
{    
    FragColor = texture(diffuse, TexCoords);
}
)";

        RenderModel model(OpenGL_ShaderProgram::FromBuffers(vertex_shader, fragment_shader));
        model._meshes = std::move(meshes);
        const unsigned shader_hanle = model._shader._id;
        glUseProgram(shader_hanle);
        model._diffuse.texture_unit = 1;
        model._diffuse.location = glGetUniformLocation(shader_hanle, "diffuse");
        model._projection_ptr = glGetUniformLocation(shader_hanle, "projection");
        model._view_ptr = glGetUniformLocation(shader_hanle, "view");
        model._model_ptr = glGetUniformLocation(shader_hanle, "model");
        model._transforms_ptr = glGetUniformLocation(shader_hanle, "bone_transforms");
        assert(model._diffuse.location >= 0);
        assert(model._projection_ptr >= 0);
        assert(model._view_ptr >= 0);
        assert(model._model_ptr >= 0);
        assert(model._transforms_ptr >= 0);
        return model;
    }

    void draw(const std::vector<glm::mat4>& transforms
        , glm::mat4 projection
        , glm::mat4 view
        , glm::mat4 model)
    {
        assert(Animation::kMaxBonesCount == transforms.size());
        glUseProgram(_shader._id);
        glUniformMatrix4fv(_projection_ptr, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(_view_ptr, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(_model_ptr, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(_transforms_ptr, GLsizei(transforms.size()), GL_FALSE, glm::value_ptr(transforms[0]));

        for (RenderMesh& mesh : _meshes)
        {
            mesh.draw(_diffuse);
        }
    }

private:
    explicit RenderModel(OpenGL_ShaderProgram&& shader)
        : _shader(std::move(shader))
    {
    }

    OpenGL_ShaderProgram _shader;
    std::vector<RenderMesh> _meshes;
    TextureLocation _diffuse;
    int _projection_ptr = -1;
    int _view_ptr = -1;
    int _model_ptr = -1;
    int _transforms_ptr = -1;
};

static std::vector<RenderMesh> OpenGL_ToRenderMesh(std::vector<AnimMesh>&& anim_meshes
    , std::initializer_list<TextureType> load_textures)
{
    auto get_by_type = [](AnimMesh& mesh, TextureType type) -> const AnimTexture&
    {
        auto it = std::find_if(mesh.textures.begin(), mesh.textures.end()
            , [&](const AnimTexture& t)
        {
            return (t.type == type);
        });
        assert(it != mesh.textures.end() && "Required texture was not found.");
        return *it;
    };

    std::vector<RenderMesh> meshes;
    for (AnimMesh& anim_mesh : anim_meshes)
    {
        std::vector<RenderTexture> textures;
        for (TextureType required : load_textures)
        {
            const AnimTexture t = get_by_type(anim_mesh, required);
            textures.push_back(OpenGL_LoadTexture(t));
        }
        meshes.push_back(RenderMesh::FromMemory(std::move(anim_mesh.vertices)
            , std::move(anim_mesh.indices), std::move(textures)));
    }
    return meshes;
}

struct AssimpOpenGL_Model
{
    RenderModel model;
    Animation animation;
};

static AssimpOpenGL_Model AssimpOpenGL_LoadAnimatedMode(const std::filesystem::path& model_path)
{
    Assimp::Importer importer;
    // As per AnimVertex.
    (void)importer.SetPropertyInteger(AI_CONFIG_PP_LBW_MAX_WEIGHTS, 4);
    const aiScene* scene = importer.ReadFile(model_path.string()
        , aiProcess_Triangulate
        | aiProcess_FlipUVs
        | aiProcess_LimitBoneWeights);
    assert(scene);
    assert((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == 0);
    assert(scene->mRootNode);

    BoneInfoRemap bone_info;
    std::vector<AnimMesh> anim_meshes = Assimp_LoadModelMeshWithAnimationsWeights(model_path, *scene, bone_info);
    Animation animation = Assimp_LoadAnimation(*scene, bone_info);
    // Load only diffuse textures, since nothing else is used.
    auto meshes = OpenGL_ToRenderMesh(std::move(anim_meshes), {TextureType::Diffuse});
    return AssimpOpenGL_Model{RenderModel::Make_NoLighting(std::move(meshes)), std::move(animation)};
}

///////////////////////////////////////////////////////////////////////////////
// APPLICATION.
struct FreeCamera
{
    glm::vec3 _position = glm::vec3(0.f);
    glm::vec3 _front = glm::vec3(0.f, 0.f, -1.f);
    glm::vec3 _up = glm::vec3(0.f, 1.f, 0.f);
    glm::vec3 _right = glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 _world_up = glm::vec3(0.f, 1.f, 0.f);
    float _yaw = -90.f;
    float _pitch = 0.f;
    float _movement_speed = 2.5f;
    float _mouse_sensitivity = 0.1f;
    float _zoom = 45.f;

#if !defined(NAN)
#  error Compiler does not support float's NAN.
#endif
    float _mouse_last_x = NAN;
    float _mouse_last_y = NAN;

    glm::mat4 view_matrix() const
    {
        return glm::lookAt(_position, _position + _front, _up);
    }

    void on_keyboard_move(glm::vec3 delta, float deltaTime)
    {
        float velocity = _movement_speed * deltaTime;
        _position += delta * velocity;
    }

    void on_mouse_scroll(float yoffset)
    {
        _zoom -= yoffset;
    }

    void on_mouse_move(float x, float y)
    {
        if (std::isnan(_mouse_last_x) || std::isnan(_mouse_last_y))
        {
            _mouse_last_x = x;
            _mouse_last_y = y;
        }
        const float xoffset = (x - _mouse_last_x) * _mouse_sensitivity;
        const float yoffset = (_mouse_last_y - y) * _mouse_sensitivity;
        _mouse_last_x = x;
        _mouse_last_y = y;
        _yaw += xoffset;
        _pitch += yoffset;
        force_refresh();
    }

    void force_refresh()
    {
        _pitch = std::clamp(_pitch, -89.0f, 89.0f);

        _front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
        _front.y = sin(glm::radians(_pitch));
        _front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
        _front = glm::normalize(_front);
        _right = glm::normalize(glm::cross(_front, _world_up));
        _up = glm::normalize(glm::cross(_right, _front));
    }
};

struct AppState
{
    FreeCamera camera = FreeCamera(glm::vec3(0.0f, 1.0f, 3.0f));
    float _dt = 0.f;
    float _last_frame_time = 0.f;
    int screen_width = 800;
    int screen_height = 600;
};

static void OnWindowResize(GLFWwindow* window, int width, int height)
{
    AppState* app = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    assert(app);
    app->screen_width = width;
    app->screen_height = height;

    glViewport(0, 0, width, height);
}

static void OnMouseMove(GLFWwindow* window, double xpos, double ypos)
{
    AppState* app = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    assert(app);
    app->camera.on_mouse_move(float(xpos), float(ypos));
}

static void OnMouseScroll(GLFWwindow* window, double xoffset, double yoffset)
{
    (void)xoffset;
    AppState* app = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    assert(app);
    app->camera.on_mouse_scroll(float(yoffset));
}

static void HandleInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
        return;
    }
    AppState* app = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    assert(app);
    FreeCamera& camera = app->camera;

    const struct
    {
        int key;
        glm::vec3 delta;
    } key_delta[] =
    {
        {GLFW_KEY_W, +camera._front},
        {GLFW_KEY_S, -camera._front},
        {GLFW_KEY_D, +camera._right},
        {GLFW_KEY_A, -camera._right},
        {GLFW_KEY_E, +camera._up},
        {GLFW_KEY_Q, -camera._up},
    };
    for (const auto& [key, delta] : key_delta)
    {
        if (glfwGetKey(window, key) == GLFW_PRESS)
        {
            camera.on_keyboard_move(delta, app->_dt);
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    assert(argc >= 2 && "app.exe <path to model to load>");
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    AppState app;
    GLFWwindow* window = glfwCreateWindow(app.screen_width, app.screen_height, "App", nullptr, nullptr);
    assert(window && "Failed to create window.");
    glfwSetWindowUserPointer(window, &app);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, OnWindowResize);
    glfwSetCursorPosCallback(window, OnMouseMove);
    glfwSetScrollCallback(window, OnMouseScroll);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    assert(gladLoadGL());

    glEnable(GL_DEPTH_TEST);
#if (0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif

    const char* const path = argv[1];
    auto [render_model, animation] = AssimpOpenGL_LoadAnimatedMode(path);

    app.camera.force_refresh();
    while (!glfwWindowShouldClose(window))
    {
        const float current_time = float(glfwGetTime());
        app._dt = current_time - app._last_frame_time;
        app._last_frame_time = current_time;
        HandleInput(window);

        animation.update(app._dt);

        if (app.screen_height <= 0)
        {
            continue; // App is minimized.
        }

        const glm::mat4 projection = glm::perspective(
            glm::radians(app.camera._zoom) // FOVY.
            , ((app.screen_width * 1.f) / app.screen_height) // Aspect.
            , 0.1f     // Near.
            , 100.0f); // Far.
        const glm::mat4 view = app.camera.view_matrix();
        const glm::mat4 model = glm::mat4(1.0f);

        glClearColor(1.f, 1.f, 1.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render_model.draw(animation.transforms(), projection, view, model);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}
