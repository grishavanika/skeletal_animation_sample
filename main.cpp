// Skeletal Animation example from
// https://learnopengl.com/Guest-Articles/2020/Skeletal-Animation.
// Refactored and simplified.
// 
// `_global_inverse` fix comes from https://ogldev.org/www/tutorial38/tutorial38.html.
// See `m_GlobalInverseTransform`.
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
#include <span>
#include <vector>
#include <optional>
#include <iterator>
#include <algorithm>
#include <filesystem>
#include <charconv>

#include <cmath>
#include <cstdio>

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
    glm::vec3 tangent;
    glm::vec3 bitangent;
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
    glm::mat4 _local_space_to_bone = glm::mat4(1.f);
    std::vector<KeyPosition> _positions;
    std::vector<KeyRotation> _rotations;
    std::vector<KeyScale> _scales;

    // "Optimization". Remember previous frame's state
    // to find next frame keys "faster". Can be removed.
    float _prev_animation_time = -1;
    int _prev_position_index = -1;
    int _prev_rotation_index = -1;
    int _prev_scale_index = -1;

    glm::mat4 interpolate_frames_at(float animation_time)
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
    glm::mat4 bone_transform;

    // Constant data.
    int parent = -1;
    glm::mat4 node_transform;
};

class Animation
{
public:
    // Limit from Vertex shader.
    static constexpr std::size_t kMaxBonesCount = 100;

    // Not actually needed. For debug purpose & simplicity.
    explicit Animation()
        : Animation(glm::mat4(1.f), {}, 0, 0.f, 0.f)
    {
    }
    explicit Animation(glm::mat4 root_inverse, std::vector<AnimNode>&& nodes, unsigned bones_count, float duration, float ticks_per_second)
        : _global_inverse(root_inverse)
        , _transforms(kMaxBonesCount, glm::mat4(1.0f))
        , _nodes(std::move(nodes))
        , _bones_count(bones_count)
        , _current_time(0.f)
        , _duration(duration)
        , _ticks_per_second(ticks_per_second)
    {
        assert(bones_count <= kMaxBonesCount);
    }

    void update(float dt)
    {
        _current_time += _ticks_per_second * dt;
        _current_time = fmod(_current_time, _duration);

        for (std::size_t i = 0, count = _nodes.size(); i < count; ++i)
        {
            AnimNode& node = _nodes[i];
            assert(int(i) > node.parent);

            const glm::mat4 bone_transform = node.bone
                ? node.bone->interpolate_frames_at(_current_time)
                : node.node_transform;
            const glm::mat4 parent_transform = (node.parent >= 0)
                ? _nodes[node.parent].bone_transform
                : glm::mat4(1.0f);
            const glm::mat4 final_ = parent_transform * bone_transform;
            node.bone_transform = final_;

            if (node.bone)
            {
                const std::size_t bone_index = node.bone->_bone_index;
                assert(bone_index < _transforms.size()
                    && "Too many bones. See kMaxBonesCount limit.");
                _transforms[bone_index] =
                    _global_inverse * final_ * node.bone->_local_space_to_bone;
            }
        }
    }

    std::span<const glm::mat4> transforms() const
    {
        if (_bones_count > 0)
        {
            // Valid animation. Return what was updated.
            return std::span<const glm::mat4>(_transforms.cbegin(), _bones_count);
        }
        // For debug purpose.
        // Not valid animation data. Return identity transforms we have.
        return _transforms;
    }

private:
    glm::mat4 _global_inverse;
    std::vector<glm::mat4> _transforms;
    std::vector<AnimNode> _nodes;
    unsigned _bones_count;
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
    TextureType type = TextureType::Invalid;
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
    glm::mat4 local_space_to_mesh;
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
        assert(mesh.mTangents);
        assert(mesh.mBitangents);
        vertices.push_back(AnimVertex{});
        AnimVertex& v = vertices.back();
        v.position = Vec_ToGLM(mesh.mVertices[i]);
        v.normal = Vec_ToGLM(mesh.mNormals[i]);
        v.texture_uv = glm::vec2(uvs[i].x, uvs[i].y);
        v.tangent = Vec_ToGLM(mesh.mTangents[i]);
        v.bitangent = Vec_ToGLM(mesh.mBitangents[i]);
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

#if (1) // STUPID HACK. Assimp & COLLADA do not see normal map.
    auto normal_it = std::find_if(textures.cbegin(), textures.cend()
        , [](const AnimTexture& t) { return t.type == TextureType::Normal; });
    auto diffuse_it = std::find_if(textures.cbegin(), textures.cend()
        , [](const AnimTexture& t) { return t.type == TextureType::Diffuse; });
    if ((normal_it == textures.cend())
        && (diffuse_it != textures.cend()))
    {
        // Don't see Normal texture, but there is Diffuse one.
        // Try to guess Normal file's path.
        auto name = diffuse_it->file_path.filename().string();
        const char kDiffusePart[] = "diffuse";
        const auto p = name.find(kDiffusePart);
        if (p != name.npos)
        {
            name.replace(p, std::size(kDiffusePart) - 1, "normal");
        }
        AnimTexture t;
        t.file_path = diffuse_it->file_path;
        t.file_path.replace_filename(name);
        t.type = TextureType::Normal;
        if (std::filesystem::exists(t.file_path))
        {
            textures.push_back(t);
        }
    }
#endif

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
    bone._local_space_to_bone = bone_info.local_space_to_mesh;

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
    if (scene.mNumAnimations == 0)
    {
        std::fprintf(stderr, "Assimp scene does not have animations. Loading invalid one (no-op).\n");
        return Animation();
    }
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
        node.node_transform = Matrix_RowToColumn(data.src->mTransformation);
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

    const glm::mat4 root = Matrix_RowToColumn(scene.mRootNode->mTransformation);
    const unsigned bones_count = unsigned(bone_info._name_to_info.size());
    return Animation(glm::inverse(root), std::move(nodes), bones_count, duration, ticks_per_second);
}

///////////////////////////////////////////////////////////////////////////////
// RENDER.
struct RenderTexture
{
    bool _loaded = false;
    unsigned texture_name = 0;
    TextureType type = TextureType::Invalid;
    explicit RenderTexture() = default;

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
    static RenderTexture Invalid_White()
    {
        unsigned data = 0xffffffff;
        return RenderTexture::FromMemory(TextureType::Invalid, GL_RGB, 1, 1, &data);
    }
    ~RenderTexture() noexcept
    {
        if (std::exchange(_loaded, false))
        {
            glDeleteTextures(1, &texture_name);
        }
    }
    RenderTexture(RenderTexture&& rhs) noexcept
        : _loaded(std::exchange(rhs._loaded, false))
        , texture_name(std::exchange(rhs.texture_name, 0))
        , type(std::exchange(rhs.type, TextureType::Invalid)) { }
    RenderTexture& operator=(RenderTexture&&) = delete;
    RenderTexture(const RenderTexture&) = delete;
    RenderTexture& operator=(const RenderTexture&) = delete;
private:
    explicit RenderTexture(unsigned name, TextureType type)
        : _loaded(true), texture_name(name), type(type)
    {
    }
};

using TextureHandle = int;

struct TexturesDB
{
    RenderTexture _invalid = RenderTexture::Invalid_White();
    std::vector<RenderTexture> _textures;

    TextureHandle add(RenderTexture&& texture)
    {
        _textures.push_back(std::move(texture));
        return TextureHandle(_textures.size() - 1);
    }

    unsigned get(TextureHandle handle) const
    {
        if ((handle < 0) || (handle >= int(_textures.size())))
        {
            return _invalid.texture_name;
        }
        return _textures[std::size_t(handle)].texture_name;
    }
};

class RenderMesh
{
public:
    static RenderMesh FromMemory(
          std::vector<AnimVertex>&& vertices
        , std::vector<unsigned>&& indices
        , TextureHandle diffuse
        , TextureHandle normal)
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
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 3/*vec3*/, GL_FLOAT, GL_FALSE
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, tangent));
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 3/*vec3*/, GL_FLOAT, GL_FALSE
            , sizeof(AnimVertex), (void*)offsetof(AnimVertex, bitangent));
        glBindVertexArray(0);

        return RenderMesh(VAO, VBO, EBO
            , indices.size(), diffuse, normal);
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
        , _normal(std::move(rhs._normal))
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

    void draw(TexturesDB& textures, int diffuse_ptr, int normal_ptr)
    {
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(diffuse_ptr, 0);
        glBindTexture(GL_TEXTURE_2D, textures.get(_diffuse));
        glActiveTexture(GL_TEXTURE1);
        glUniform1i(normal_ptr, 1);
        glBindTexture(GL_TEXTURE_2D, textures.get(_normal));

        glBindVertexArray(_VAO);
        glDrawElements(GL_TRIANGLES, GLsizei(_indicies_count), GL_UNSIGNED_INT, 0);
    }

private:
    explicit RenderMesh(unsigned VAO, unsigned VBO, unsigned EBO
        , std::size_t indicies_count, TextureHandle diffuse, TextureHandle normal)
        : _VAO(VAO)
        , _VBO(VBO)
        , _EBO(EBO)
        , _indicies_count(indicies_count)
        , _diffuse(diffuse)
        , _normal(normal)
    {
    }
    unsigned _VAO;
    unsigned _VBO;
    unsigned _EBO;
    std::size_t _indicies_count;
    TextureHandle _diffuse;
    TextureHandle _normal;
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
    static RenderModel Make_SimpleNormalMapping(TexturesDB&& textures, std::vector<RenderMesh>&& meshes)
    {
        const char* vertex_shader = R"(
#version 330 core

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_UV;
layout(location = 3) in ivec4 in_BoneIds;
layout(location = 4) in vec4 in_Weights;
layout(location = 5) in vec3 in_Tangent;
layout(location = 6) in vec3 in_Bitangent;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

// See kMaxBonesCount.
uniform mat4 bone_transforms[100];

out vec2 v_UV;
out vec3 v_Position;
out vec3 v_Normal;
out mat3 v_TBN;

void main()
{
    mat4 S = mat4(0.f);
    for (int i = 0; i < 4; ++i)
    {
        if (in_BoneIds[i] >= 0)
        {
            S += (bone_transforms[in_BoneIds[i]] * in_Weights[i]);
        }
    }
    if (in_BoneIds[0] < 0)
    {
        // In case vertex has no any bone.
        // For debug purpose, make it visible.
        S = mat4(1.f);
    }
    mat4 MVP = projection * view * model;
    gl_Position = MVP * S * vec4(in_Position, 1.f);
    v_Position = vec3(model * S * vec4(in_Position, 1.f));
    v_UV = in_UV;
    
    mat3 MS = mat3(model * S);
    vec3 T = normalize(MS * in_Tangent);
    vec3 B = normalize(MS * in_Bitangent);
    vec3 N = normalize(MS * in_Normal);
    v_TBN = mat3(T, B, N);
    v_Normal = N;
}
)";

        const char* fragment_shader = R"(
#version 330 core

uniform sampler2D diffuse_sampler;
uniform sampler2D normal_sampler;
uniform vec3 light_position;
uniform vec3 view_position;

in vec2 v_UV;
in vec3 v_Position;
in vec3 v_Normal;
in mat3 v_TBN;

out vec4 _Color;

void main()
{
    vec3 light_color = vec3(1.f, 1.f, 1.f);
    float abbient_K = 0.6f;
    float specular_K = 1.2f;
    float specular_P = 100f;

    // Ambient.
    vec3 ambient = abbient_K * light_color;
    
    // Diffuse.
    vec3 N = vec3(texture(normal_sampler, v_UV));
    N = N * 2.0 - 1.0;
    N = normalize(v_TBN * N);

    vec3 light_dir = normalize(light_position - v_Position);
    float diff = max(dot(N, light_dir), 0.f);
    vec3 diffuse = diff * light_color;
    
    // Specular.
    vec3 view_dir = normalize(view_position - v_Position);
    vec3 reflect_dir = reflect(-light_dir, N);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.f), specular_P);
    vec3 specular = specular_K * spec * light_color;
    
    vec3 object_color = vec3(texture(diffuse_sampler, v_UV));
    vec3 color = (ambient + diffuse + specular) * object_color;
    _Color = vec4(color, 1.f);
}
)";
        auto shader = OpenGL_ShaderProgram::FromBuffers(vertex_shader, fragment_shader);
        RenderModel model(std::move(textures), std::move(shader));
        model._meshes = std::move(meshes);
        const unsigned shader_hanle = model._shader._id;
        glUseProgram(shader_hanle);
        model._diffuse_ptr = glGetUniformLocation(shader_hanle, "diffuse_sampler");
        model._normal_ptr = glGetUniformLocation(shader_hanle, "normal_sampler");
        model._projection_ptr = glGetUniformLocation(shader_hanle, "projection");
        model._view_ptr = glGetUniformLocation(shader_hanle, "view");
        model._model_ptr = glGetUniformLocation(shader_hanle, "model");
        model._transforms_ptr = glGetUniformLocation(shader_hanle, "bone_transforms");
        model._light_position_ptr = glGetUniformLocation(shader_hanle, "light_position");
        model._view_position_ptr = glGetUniformLocation(shader_hanle, "view_position");
        assert(model._diffuse_ptr >= 0);
        assert(model._normal_ptr >= 0);
        assert(model._projection_ptr >= 0);
        assert(model._view_ptr >= 0);
        assert(model._model_ptr >= 0);
        assert(model._transforms_ptr >= 0);
        assert(model._light_position_ptr >= 0);
        assert(model._view_position_ptr >= 0);
        return model;
    }

    void draw(std::span<const glm::mat4> transforms
        , glm::mat4 projection
        , glm::mat4 view
        , glm::mat4 model
        , glm::vec3 light_position
        , glm::vec3 view_position)
    {
        assert(transforms.size() > 0);
        assert(transforms.size() <= Animation::kMaxBonesCount);
        glUseProgram(_shader._id);
        glUniformMatrix4fv(_projection_ptr, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(_view_ptr, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(_model_ptr, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(_transforms_ptr, GLsizei(transforms.size()), GL_FALSE, glm::value_ptr(transforms[0]));
        glUniform3fv(_light_position_ptr, 1, glm::value_ptr(light_position));
        glUniform3fv(_view_position_ptr, 1, glm::value_ptr(view_position));

        for (RenderMesh& mesh : _meshes)
        {
            mesh.draw(_textures, _diffuse_ptr, _normal_ptr);
        }
    }

private:
    explicit RenderModel(TexturesDB&& textures, OpenGL_ShaderProgram&& shader)
        : _textures(std::move(textures))
        , _shader(std::move(shader))
    {
    }

    TexturesDB _textures;
    OpenGL_ShaderProgram _shader;
    std::vector<RenderMesh> _meshes;
    int _diffuse_ptr = -1;
    int _normal_ptr = -1;
    int _projection_ptr = -1;
    int _view_ptr = -1;
    int _model_ptr = -1;
    int _transforms_ptr = -1;
    int _light_position_ptr = -1;
    int _view_position_ptr = -1;
};

static std::vector<RenderMesh> OpenGL_LoadRenderMesh(TexturesDB& textures
    , std::vector<AnimMesh>&& anim_meshes)
{
    auto load_by_type = [&](AnimMesh& mesh, TextureType type, const char* debug_name)
    {
        auto it = std::find_if(mesh.textures.begin(), mesh.textures.end()
            , [&](const AnimTexture& t)
        {
            return (t.type == type);
        });
        if (it != mesh.textures.end())
        {
            return textures.add(OpenGL_LoadTexture(*it));
        }
        std::fprintf(stderr, "Mesh is missing '%s' texture.\n", debug_name);
        return TextureHandle(-1);
    };

    std::vector<RenderMesh> meshes;
    for (AnimMesh& anim_mesh : anim_meshes)
    {
        meshes.push_back(RenderMesh::FromMemory(
              std::move(anim_mesh.vertices)
            , std::move(anim_mesh.indices)
            , load_by_type(anim_mesh, TextureType::Diffuse, "diffuse")
            , load_by_type(anim_mesh, TextureType::Normal, "normal")));
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
        | aiProcess_CalcTangentSpace
        | aiProcess_FlipUVs
        | aiProcess_LimitBoneWeights);
    assert(scene);
    assert((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == 0);
    assert(scene->mRootNode);

    BoneInfoRemap bone_info;
    TexturesDB textures;
    std::vector<AnimMesh> anim_meshes =
        Assimp_LoadModelMeshWithAnimationsWeights(model_path, *scene, bone_info);
    Animation animation =
        Assimp_LoadAnimation(*scene, bone_info);
    std::vector<RenderMesh> meshes =
        OpenGL_LoadRenderMesh(textures, std::move(anim_meshes));
    RenderModel model = 
        RenderModel::Make_SimpleNormalMapping(std::move(textures), std::move(meshes));
    return AssimpOpenGL_Model{std::move(model), std::move(animation)};
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
    bool _pause_animation = false;
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

static void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;
    if ((key == GLFW_KEY_SPACE) && (action == GLFW_PRESS))
    {
        AppState* app = static_cast<AppState*>(glfwGetWindowUserPointer(window));
        assert(app);
        app->_pause_animation ^= true;
    }
}

static float Get_ArgFloat(int argc, char* argv[], const char* name, float or_default = -1.f)
{
    int index = -1;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string_view(argv[i]) == name)
        {
            index = (i + 1);
            break;
        }
    }
    if ((index < 0) || (index >= argc))
    {
        return or_default;
    }
    float v = or_default;
    const char* v_str = argv[index];
    const char* v_end = v_str + strlen(v_str);
    (void)std::from_chars(v_str, v_end, v);
    return v;
}

int main(int argc, char* argv[])
{
    assert(argc >= 2 && "app.exe <path to model to load>");
    const char* const path = argv[1];
    // Models from mixamo.com are HUGE, make them smaller.
    const float model_scale = Get_ArgFloat(argc, argv, "--scale", 0.012f);
    std::fprintf(stdout, "Model to load: %s.\n", path);
    std::fprintf(stdout, "Model scale: %f.\n", model_scale);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    AppState app;
    GLFWwindow* window = glfwCreateWindow(app.screen_width, app.screen_height, "App", nullptr, nullptr);
    assert(window && "Failed to create window.");
    glfwSetWindowUserPointer(window, &app);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, OnWindowResize);
    glfwSetCursorPosCallback(window, OnMouseMove);
    glfwSetScrollCallback(window, OnMouseScroll);
    glfwSetKeyCallback(window, OnKeyEvent);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    const int glad_ok = gladLoadGL();
    assert(glad_ok > 0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
#if (0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif

    auto [render_model, animation] = AssimpOpenGL_LoadAnimatedMode(path);

    app.camera.force_refresh();
    while (!glfwWindowShouldClose(window))
    {
        const float current_time = float(glfwGetTime());
        app._dt = current_time - app._last_frame_time;
        app._last_frame_time = current_time;
        HandleInput(window);

        if (!app._pause_animation)
        {
            animation.update(app._dt);
        }

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
        glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(model_scale));

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render_model.draw(animation.transforms()
            , projection, view, model
            , glm::vec3(0.0f, 1.0f, 3.0f) // Light position.
            , app.camera._position);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}
