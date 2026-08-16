// Tests the public Skills catalog/read handlers through their constructed
// handler interface (server_http_req in, server_http_res out) without binding
// any HTTP route. This is the narrow direct-handler verification seam for the
// Skills contract: the fixture drives real request headers and request JSON
// through server_skills::handle_get / handle_post, exactly as the HTTP layer
// would after deserializing a request.
//
#include "../tools/server/server-skills.h"

#include "common.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

static int g_failures = 0;

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__    \
                      << ": " #condition << "\n";                             \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

static std::vector<fs::path> g_temp_roots;

static fs::path make_temp_dir() {
    static size_t counter = 0;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
        ("test-server-skills-" + std::to_string(now) + "-" + std::to_string(counter++));
    fs::create_directories(root);
    g_temp_roots.push_back(root);
    return root;
}

static void write_bytes(const fs::path & path, const std::string & bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
}

static fs::path write_skill(const fs::path & root, const std::string & provider, const std::string & name,
                            const std::string & body = "Use <safe> & sound.",
                            const std::string & description = "Use <safe> & sound.") {
    const fs::path skill = root / ("." + provider) / "skills" / name;
    fs::create_directories(skill);
    std::ofstream out(skill / "SKILL.md", std::ios::binary);
    out << "---\n"
        << "name: " << name << "\n"
        << "description: " << description << "\n"
        << "license: Apache-2.0\n"
        << "compatibility: Requires git\n"
        << "allowed-tools: Bash(git:*) Read\n"
        << "metadata:\n"
        << "  author: test-suite\n"
        << "---\n"
        << body;
    return skill;
}

static const std::function<bool()> should_stop = []() { return false; };

static server_http_res_ptr do_get(server_skills & skills, const std::map<std::string, std::string> & headers = {}) {
    const server_http_req request{{}, headers, "/skills", "", "", {}, should_stop};
    return skills.handle_get(request);
}

static server_http_res_ptr do_post(server_skills & skills, const std::string & body,
                                   const std::map<std::string, std::string> & headers = {}) {
    const server_http_req request{{}, headers, "/skills/read", "", body, {}, should_stop};
    return skills.handle_post(request);
}

static json parse_body(const server_http_res_ptr & res) {
    return json::parse(res->data);
}

// Mirrors the server launched with HOME=<home> and process CWD=<project>: the
// service captures both at construction and re-resolves per request. Requests
// without an X-Skill-Cwd header therefore resolve against the fixture project,
// and requests with the header must produce the identical catalog. An optional
// token callback lets cache tests observe measurement through public results.
static server_skills make_skills(const fs::path & home, const fs::path & project, bool trust_project_skills,
                                 std::vector<std::string> providers = {"claude"},
                                 token_count_callback count_tokens = {}) {
    common_set_env("HOME", home.string());
    fs::current_path(project);
    return server_skills(server_skills_config{true, trust_project_skills, std::move(providers)}, std::move(count_tokens));
}

static void test_catalog_global_only() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);
    write_skill(home, "agents", "global-only");
    write_skill(home, "agents", "shared", "global body");
    write_skill(project, "agents", "shared", "project body");
    write_skill(project, "claude", "provider-only");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ false);
    const server_http_res_ptr response = do_get(skills);
    CHECK(response != nullptr);
    CHECK(response->status == 200);
    const json body = parse_body(response);
    CHECK(body.at("skills").size() == 2);
    CHECK(body.at("skills")[0].at("name") == "global-only");
    CHECK(body.at("skills")[1].at("name") == "shared");
    for (const auto & skill : body.at("skills")) {
        CHECK(skill.at("scope") == "global");
    }
    const std::string payload = body.dump();
    CHECK(payload.find(home.string()) == std::string::npos);
    CHECK(payload.find(project.string()) == std::string::npos);
}

static void test_catalog_precedence_and_cwd_identity() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);
    write_skill(home, "agents", "global-only");
    write_skill(home, "agents", "shared", "global body");
    write_skill(project, "agents", "shared", "project body");
    write_skill(project, "claude", "provider-only");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr implicit = do_get(skills);
    const server_http_res_ptr explicit_req = do_get(skills, {{"X-Skill-Cwd", project.string()}});
    CHECK(implicit != nullptr && explicit_req != nullptr);
    CHECK(implicit->status == 200);
    CHECK(explicit_req->status == 200);
    const json implicit_body = parse_body(implicit);
    const json explicit_body = parse_body(explicit_req);
    CHECK(implicit_body.at("skills") == explicit_body.at("skills"));
    // project .agents beats global .agents; the other roots follow in order
    CHECK(implicit_body.at("skills").size() == 3);
    CHECK(implicit_body.at("skills")[0].at("name") == "shared");
    CHECK(implicit_body.at("skills")[0].at("scope") == "project");
    CHECK(implicit_body.at("skills")[0].at("provider") == "agents");
    CHECK(implicit_body.at("skills")[1].at("name") == "provider-only");
    CHECK(implicit_body.at("skills")[1].at("scope") == "project");
    CHECK(implicit_body.at("skills")[1].at("provider") == "claude");
    CHECK(implicit_body.at("skills")[2].at("name") == "global-only");
    CHECK(implicit_body.at("skills")[2].at("scope") == "global");
    const json shadows = implicit_body.at("diagnostics");
    CHECK(shadows.size() == 1);
    CHECK(shadows[0].at("severity") == "warning");
    CHECK(shadows[0].at("code") == "skill_shadowed");
    CHECK(shadows[0].at("name") == "shared");
    CHECK(shadows[0].at("scope") == "global");
    CHECK(shadows[0].at("provider") == "agents");
    CHECK(shadows[0].at("message") == "Skill is shadowed by a higher-precedence entry");
}

static void test_catalog_rejects_invalid_cwd_and_escapes_xml() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);
    write_skill(home, "agents", "global-only");
    write_skill(home, "agents", "shared", "global body");
    write_skill(project, "agents", "shared", "project body");
    write_skill(project, "claude", "provider-only");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr invalid = do_get(skills, {{"X-Skill-Cwd", (tmp / "missing").string()}});
    CHECK(invalid != nullptr);
    CHECK(invalid->status == 400);

    const server_http_res_ptr response = do_get(skills);
    CHECK(response != nullptr);
    CHECK(response->status == 200);
    const json body = parse_body(response);
    bool found = false;
    for (const auto & skill : body.at("skills")) {
        if (skill.at("name") == "shared") {
            found = true;
            const std::string catalog_xml = skill.at("catalog_xml").get<std::string>();
            CHECK(catalog_xml.find("&lt;safe&gt; &amp; sound.") != std::string::npos);
        }
    }
    CHECK(found);
    const std::string payload = body.dump();
    CHECK(payload.find(home.string()) == std::string::npos);
    CHECK(payload.find(project.string()) == std::string::npos);
}

static void test_catalog_diagnoses_invalid_candidates() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path oversized = write_skill(home, "agents", "oversized");
    write_bytes(oversized / "SKILL.md", std::string(1024 * 1024 + 1, 'x'));
    const fs::path invalid_utf8 = write_skill(home, "agents", "invalid-utf8");
    write_bytes(invalid_utf8 / "SKILL.md", "---\ndescription: x\n---\n\xff");
    const fs::path missing_description = write_skill(home, "agents", "missing-description");
    write_bytes(missing_description / "SKILL.md", "---\nname: missing-description\n---\nbody");
    const fs::path malformed_frontmatter = write_skill(home, "agents", "malformed-frontmatter");
    write_bytes(malformed_frontmatter / "SKILL.md", "---\nname: malformed-frontmatter\ndescription: desc\nbody without closing delimiter");
    // An unsafe directory basename (contains a backslash) must be diagnosed but
    // must never surface as a diagnostic name. Its SKILL.md is a directory, so
    // the candidate is rejected with skill_unreadable after the canonical checks.
    const fs::path unsafe_basename = home / ".agents" / "skills" / "bad\\name";
    fs::create_directories(unsafe_basename);
    fs::create_directories(unsafe_basename / "SKILL.md");
    // A safe-named symlink candidate escaping the configured root is rejected
    // with skill_unsafe_path and identified by its directory basename.
    const fs::path outside = tmp / "outside";
    fs::create_directories(outside);
    write_bytes(outside / "SKILL.md", "---\ndescription: secret\n---\nbody");
    std::error_code link_ec;
    fs::create_directory_symlink(outside, home / ".agents" / "skills" / "escape", link_ec);

    server_skills skills = make_skills(home, project, /* trust_project_skills */ false);
    const server_http_res_ptr response = do_get(skills);
    CHECK(response != nullptr);
    CHECK(response->status == 200);
    const json body = parse_body(response);
    std::map<std::string, json> diagnostics;
    for (const auto & item : body.at("diagnostics")) {
        diagnostics[item.at("code").get<std::string>()] = item;
    }
    CHECK(diagnostics.count("skill_invalid_frontmatter") == 1);
    CHECK(diagnostics.count("skill_too_large") == 1);
    CHECK(diagnostics.count("skill_invalid_utf8") == 1);
    CHECK(diagnostics.count("skill_missing_description") == 1);
    CHECK(diagnostics.count("skill_unreadable") == 1);
    // Each safe candidate diagnostic carries its directory name and the global
    // agents identity; the unsafe basename stays empty.
    CHECK(diagnostics.at("skill_invalid_frontmatter").at("name") == "malformed-frontmatter");
    CHECK(diagnostics.at("skill_invalid_frontmatter").at("scope") == "global");
    CHECK(diagnostics.at("skill_invalid_frontmatter").at("provider") == "agents");
    CHECK(diagnostics.at("skill_too_large").at("name") == "oversized");
    CHECK(diagnostics.at("skill_too_large").at("scope") == "global");
    CHECK(diagnostics.at("skill_too_large").at("provider") == "agents");
    CHECK(diagnostics.at("skill_invalid_utf8").at("name") == "invalid-utf8");
    CHECK(diagnostics.at("skill_invalid_utf8").at("scope") == "global");
    CHECK(diagnostics.at("skill_invalid_utf8").at("provider") == "agents");
    CHECK(diagnostics.at("skill_missing_description").at("name") == "missing-description");
    CHECK(diagnostics.at("skill_missing_description").at("scope") == "global");
    CHECK(diagnostics.at("skill_missing_description").at("provider") == "agents");
    CHECK(diagnostics.at("skill_unreadable").value("name", "") == "");
    CHECK(diagnostics.at("skill_unreadable").at("scope") == "global");
    CHECK(diagnostics.at("skill_unreadable").at("provider") == "agents");
    if (!link_ec) {
        // symlinks unavailable (e.g. Windows without developer mode): skip
        CHECK(diagnostics.count("skill_unsafe_path") == 1);
        CHECK(diagnostics.at("skill_unsafe_path").at("name") == "escape");
        CHECK(diagnostics.at("skill_unsafe_path").at("scope") == "global");
        CHECK(diagnostics.at("skill_unsafe_path").at("provider") == "agents");
    }
    bool malformed_loaded = false;
    for (const auto & skill : body.at("skills")) {
        malformed_loaded = malformed_loaded || skill.at("name") == "malformed-frontmatter";
    }
    CHECK(!malformed_loaded);
}

static void test_reads_current_base_and_resources() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path skill = write_skill(project, "agents", "current", "first");
    const fs::path resource = skill / "references" / "DETAILS.md";
    fs::create_directories(resource.parent_path());
    write_bytes(resource, "first resource");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr first = do_post(skills, R"({"name":"current"})");
    CHECK(first != nullptr);
    CHECK(first->status == 200);
    const json first_body = parse_body(first);
    CHECK(first_body.at("kind") == "skill");
    CHECK(first_body.at("content_xml").get<std::string>().find("first") != std::string::npos);

    // the handlers must re-resolve and re-read on every request
    write_bytes(skill / "SKILL.md", "---\nname: current\ndescription: Use <safe> & sound.\n---\nsecond");
    write_bytes(resource, "second resource");

    const server_http_res_ptr second = do_post(skills, R"({"name":"current"})");
    CHECK(second != nullptr);
    CHECK(second->status == 200);
    const json second_body = parse_body(second);
    CHECK(second_body.at("content_xml").get<std::string>().find("second") != std::string::npos);

    const server_http_res_ptr resource_response = do_post(skills, R"({"name":"current","path":"references/DETAILS.md"})");
    CHECK(resource_response != nullptr);
    CHECK(resource_response->status == 200);
    const json resource_body = parse_body(resource_response);
    CHECK(resource_body.at("kind") == "resource");
    CHECK(resource_body.at("resource").at("path") == "references/DETAILS.md");
    CHECK(resource_body.at("content_xml").get<std::string>().find("second resource") != std::string::npos);
}

static void test_read_rejects_unsafe_paths_and_ranks_suggestions() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path skill = write_skill(project, "agents", "resources");
    const fs::path references = skill / "references";
    fs::create_directories(references);
    write_bytes(references / "DETAILS.md", "details");
    write_bytes(references / "OTHER.md", "other");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr traversal = do_post(skills, R"({"name":"resources","path":"../escape"})");
    CHECK(traversal != nullptr);
    CHECK(traversal->status == 400);

    const server_http_res_ptr missing = do_post(skills, R"({"name":"resources","path":"references/DETAIL.md"})");
    CHECK(missing != nullptr);
    CHECK(missing->status == 404);
    const json missing_body = parse_body(missing);
    CHECK(missing_body.at("suggestions").size() >= 1);
    CHECK(missing_body.at("suggestions").size() <= 3);
    CHECK(missing_body.at("suggestions")[0] == "references/DETAILS.md");
}

static void test_resource_listing_bounds_and_invalid_content() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path skill = write_skill(project, "agents", "bounded");
    for (int index = 0; index < 257; ++index) {
        std::ostringstream name;
        name << "resource-" << std::setw(3) << std::setfill('0') << index << ".txt";
        write_bytes(skill / name.str(), std::to_string(index));
    }
    const fs::path nested = skill / "one" / "two" / "three" / "four" / "five.txt";
    fs::create_directories(nested.parent_path());
    write_bytes(nested, "too deep");
    write_bytes(skill / "invalid.txt", std::string("\xff", 1));

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr base = do_post(skills, R"({"name":"bounded"})");
    CHECK(base != nullptr);
    CHECK(base->status == 200);
    const json base_body = parse_body(base);
    CHECK(base_body.at("resources").at("paths").size() == 256);
    CHECK(base_body.at("resources").at("truncated") == true);
    bool found_deep = false;
    for (const auto & path : base_body.at("resources").at("paths")) {
        if (path.get<std::string>() == "one/two/three/four/five.txt") {
            found_deep = true;
        }
    }
    CHECK(!found_deep);

    const server_http_res_ptr invalid = do_post(skills, R"({"name":"bounded","path":"invalid.txt"})");
    CHECK(invalid != nullptr);
    CHECK(invalid->status == 400);
}

static void test_rejects_directory_and_file_symlink_escapes() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    const fs::path outside = tmp / "outside";
    fs::create_directories(home);
    fs::create_directories(project);
    fs::create_directories(outside);
    write_bytes(outside / "secret.txt", "secret");

    const fs::path skill = write_skill(project, "agents", "links");
    std::error_code link_ec;
    fs::create_directory_symlink(outside, skill / "escape", link_ec);
    fs::create_symlink(outside / "secret.txt", skill / "file-link.txt", link_ec);
    if (link_ec) {
        // symlinks unavailable (e.g. Windows without developer mode): skip
        return;
    }

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr catalog = do_get(skills);
    CHECK(catalog != nullptr);
    CHECK(catalog->status == 200);

    const server_http_res_ptr base = do_post(skills, R"({"name":"links"})");
    CHECK(base != nullptr);
    CHECK(base->status == 200);
    const json base_body = parse_body(base);
    bool escape_listed = false;
    bool file_link_listed = false;
    for (const auto & path : base_body.at("resources").at("paths")) {
        const std::string value = path.get<std::string>();
        if (value == "escape/secret.txt") {
            escape_listed = true;
        }
        if (value == "file-link.txt") {
            file_link_listed = true;
        }
    }
    CHECK(!escape_listed);
    CHECK(!file_link_listed);

    const server_http_res_ptr directory_escape = do_post(skills, R"({"name":"links","path":"escape/secret.txt"})");
    CHECK(directory_escape != nullptr);
    CHECK(directory_escape->status == 400);

    const server_http_res_ptr file_escape = do_post(skills, R"({"name":"links","path":"file-link.txt"})");
    CHECK(file_escape != nullptr);
    CHECK(file_escape->status == 400);
}

static void test_configured_root_symlink_containment() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    const fs::path escaped_project_root = tmp / "escaped-project";
    const fs::path escaped_home_root = tmp / "escaped-home";
    fs::create_directories(home);
    fs::create_directories(project);
    fs::create_directories(escaped_project_root / "leaked");
    fs::create_directories(escaped_home_root / "leaked-global");

    // valid skills living entirely outside the project CWD and the captured HOME
    write_bytes(escaped_project_root / "leaked" / "SKILL.md", "---\nname: leaked\ndescription: leaked\n---\nbody");
    write_bytes(escaped_home_root / "leaked-global" / "SKILL.md", "---\nname: leaked-global\ndescription: leaked\n---\nbody");

    std::error_code link_ec;
    // project provider root escapes the effective CWD base
    fs::create_directories(project / ".claude");
    fs::create_directory_symlink(escaped_project_root, project / ".claude" / "skills", link_ec);
    // global agents root escapes the captured HOME base
    fs::create_directory_symlink(escaped_home_root, home / ".agents" / "skills", link_ec);
    if (link_ec) {
        // symlinks unavailable (e.g. Windows without developer mode): skip
        return;
    }
    // a valid in-base symlink is preserved: project .agents -> project/real-skills
    const fs::path real_skills = project / "real-skills";
    fs::create_directories(real_skills / "inbase");
    write_bytes(real_skills / "inbase" / "SKILL.md", "---\nname: inbase\ndescription: inbase\n---\nbody");
    fs::create_directory_symlink(real_skills, project / ".agents" / "skills", link_ec);
    if (link_ec) {
        return;
    }

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    // GET: escaped roots are rejected with skill_root_invalid, in-base root is used
    const server_http_res_ptr catalog = do_get(skills);
    CHECK(catalog != nullptr);
    CHECK(catalog->status == 200);
    const json catalog_body = parse_body(catalog);
    CHECK(catalog_body.at("skills").size() == 1);
    CHECK(catalog_body.at("skills")[0].at("name") == "inbase");
    CHECK(catalog_body.at("skills")[0].at("provider") == "agents");
    bool saw_project_root_invalid = false;
    bool saw_global_root_invalid = false;
    for (const auto & item : catalog_body.at("diagnostics")) {
        if (item.at("code") == "skill_root_invalid" && item.at("scope") == "project" && item.at("provider") == "claude") {
            saw_project_root_invalid = true;
        }
        if (item.at("code") == "skill_root_invalid" && item.at("scope") == "global" && item.at("provider") == "agents") {
            saw_global_root_invalid = true;
        }
    }
    CHECK(saw_project_root_invalid);
    CHECK(saw_global_root_invalid);

    // POST: the same containment applies and the read still succeeds with diagnostics
    const server_http_res_ptr base = do_post(skills, R"({"name":"inbase"})");
    CHECK(base != nullptr);
    CHECK(base->status == 200);
    const json base_body = parse_body(base);
    bool read_has_root_invalid = false;
    for (const auto & item : base_body.at("diagnostics")) {
        if (item.at("code") == "skill_root_invalid") {
            read_has_root_invalid = true;
        }
    }
    CHECK(read_has_root_invalid);

    // escaped roots never publish their skills
    const server_http_res_ptr missing = do_post(skills, R"({"name":"leaked"})");
    CHECK(missing != nullptr);
    CHECK(missing->status == 404);
    const server_http_res_ptr missing_global = do_post(skills, R"({"name":"leaked-global"})");
    CHECK(missing_global != nullptr);
    CHECK(missing_global->status == 404);
}

static void test_cosmetic_name_diagnostics() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    // a safe frontmatter name violating only the cosmetic grammar (space,
    // underscore) is retained as the public name with a defect warning, never
    // replaced by the directory name
    const fs::path spaced = project / ".agents" / "skills" / "my-skill";
    fs::create_directories(spaced);
    write_bytes(spaced / "SKILL.md", "---\nname: My Skill\ndescription: desc\n---\nbody");
    const fs::path underscored = project / ".agents" / "skills" / "Bad_Name";
    fs::create_directories(underscored);
    write_bytes(underscored / "SKILL.md", "---\nname: Bad_Name\ndescription: desc\n---\nbody");
    // an unsafe frontmatter name (path separator) is rejected outright
    const fs::path unsafe = project / ".agents" / "skills" / "unsafe";
    fs::create_directories(unsafe);
    write_bytes(unsafe / "SKILL.md", "---\nname: foo/bar\ndescription: desc\n---\nbody");
    // a cosmetic skill for the POST read
    const fs::path fine = project / ".agents" / "skills" / "fine";
    fs::create_directories(fine);
    write_bytes(fine / "SKILL.md", "---\nname: fine\ndescription: desc\n---\nbody");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr catalog = do_get(skills);
    CHECK(catalog != nullptr);
    CHECK(catalog->status == 200);
    const json catalog_body = parse_body(catalog);
    CHECK(catalog_body.at("skills").size() == 3);
    bool found_spaced = false;
    bool found_underscored = false;
    bool found_fine = false;
    for (const auto & skill : catalog_body.at("skills")) {
        const std::string name = skill.at("name").get<std::string>();
        if (name == "My Skill") {
            found_spaced = true;
        }
        if (name == "Bad_Name") {
            found_underscored = true;
        }
        if (name == "fine") {
            found_fine = true;
        }
    }
    CHECK(found_spaced);
    CHECK(found_underscored);
    CHECK(found_fine);
    int defect_count = 0;
    bool saw_spaced_defect = false;
    bool saw_underscored_defect = false;
    bool saw_unnamed_defect = false;
    for (const auto & item : catalog_body.at("diagnostics")) {
        if (item.at("code") == "skill_name_invalid") {
            ++defect_count;
            if (item.contains("name") && item.at("name") == "My Skill") {
                saw_spaced_defect = true;
            }
            if (item.contains("name") && item.at("name") == "Bad_Name") {
                saw_underscored_defect = true;
            }
            if (!item.contains("name")) {
                saw_unnamed_defect = true;
            }
        }
    }
    CHECK(defect_count == 3);
    CHECK(saw_spaced_defect);
    CHECK(saw_underscored_defect);
    CHECK(saw_unnamed_defect);

    // the POST read path surfaces the same diagnostics instead of an empty array
    const server_http_res_ptr read = do_post(skills, R"({"name":"My Skill"})");
    CHECK(read != nullptr);
    CHECK(read->status == 200);
    const json read_body = parse_body(read);
    bool read_saw_defect = false;
    for (const auto & item : read_body.at("diagnostics")) {
        if (item.at("code") == "skill_name_invalid") {
            read_saw_defect = true;
        }
    }
    CHECK(read_saw_defect);

    // an unsafe name is rejected at request validation and is not selectable
    const server_http_res_ptr unsafe_read = do_post(skills, R"({"name":"foo/bar"})");
    CHECK(unsafe_read != nullptr);
    CHECK(unsafe_read->status == 400);
}

static void test_cosmetic_name_leading_trailing_hyphen() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    // leading/trailing hyphens violate the cosmetic grammar but are safe, so
    // the public frontmatter name is retained with a defect warning
    const fs::path leading = project / ".agents" / "skills" / "leading";
    fs::create_directories(leading);
    write_bytes(leading / "SKILL.md", "---\nname: -leading\ndescription: desc\n---\nbody");
    const fs::path trailing = project / ".agents" / "skills" / "trailing";
    fs::create_directories(trailing);
    write_bytes(trailing / "SKILL.md", "---\nname: trailing-\ndescription: desc\n---\nbody");
    // a fully cosmetic control
    const fs::path hyphen = project / ".agents" / "skills" / "hyphen";
    fs::create_directories(hyphen);
    write_bytes(hyphen / "SKILL.md", "---\nname: hyphen\ndescription: desc\n---\nbody");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr catalog = do_get(skills);
    CHECK(catalog != nullptr);
    CHECK(catalog->status == 200);
    const json catalog_body = parse_body(catalog);
    CHECK(catalog_body.at("skills").size() == 3);
    bool found_leading = false;
    bool found_trailing = false;
    bool found_hyphen = false;
    for (const auto & skill : catalog_body.at("skills")) {
        const std::string name = skill.at("name").get<std::string>();
        if (name == "-leading") {
            found_leading = true;
        }
        if (name == "trailing-") {
            found_trailing = true;
        }
        if (name == "hyphen") {
            found_hyphen = true;
        }
    }
    CHECK(found_leading);
    CHECK(found_trailing);
    CHECK(found_hyphen);
    int defect_count = 0;
    bool saw_leading_defect = false;
    bool saw_trailing_defect = false;
    for (const auto & item : catalog_body.at("diagnostics")) {
        if (item.at("code") == "skill_name_invalid") {
            ++defect_count;
            if (item.contains("name") && item.at("name") == "-leading") {
                saw_leading_defect = true;
            }
            if (item.contains("name") && item.at("name") == "trailing-") {
                saw_trailing_defect = true;
            }
        }
    }
    CHECK(defect_count == 2);
    CHECK(saw_leading_defect);
    CHECK(saw_trailing_defect);

    // the retained safe names resolve through POST with the same diagnostics
    const server_http_res_ptr read = do_post(skills, R"({"name":"-leading"})");
    CHECK(read != nullptr);
    CHECK(read->status == 200);
    const json read_body = parse_body(read);
    bool read_saw_defect = false;
    for (const auto & item : read_body.at("diagnostics")) {
        if (item.at("code") == "skill_name_invalid") {
            read_saw_defect = true;
        }
    }
    CHECK(read_saw_defect);
}

// Literal (`|`) and folded (`>`) block-scalar descriptions, each chomping sign
// (none = clip, `-` = strip, `+` = keep): the parser must consume the indented
// continuation, strip the common indentation, and expose the parsed text
// through GET /skills and POST /skills/read metadata - never the raw
// `|-`/`>-` indicators.
static void test_block_scalar_descriptions() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path literal = project / ".agents" / "skills" / "literal";
    fs::create_directories(literal);
    write_bytes(literal / "SKILL.md",
        "---\n"
        "name: literal\n"
        "description: |\n"
        "  First line\n"
        "  Second line\n"
        "\n"
        "  Fourth line\n"
        "license: Apache-2.0\n"
        "---\n"
        "body");
    const fs::path literal_strip = project / ".agents" / "skills" / "literal-strip";
    fs::create_directories(literal_strip);
    write_bytes(literal_strip / "SKILL.md",
        "---\n"
        "name: literal-strip\n"
        "description: |-\n"
        "  Strip line one\n"
        "  Strip line two\n"
        "\n"
        "license: Apache-2.0\n"
        "---\n"
        "body");
    const fs::path literal_keep = project / ".agents" / "skills" / "literal-keep";
    fs::create_directories(literal_keep);
    write_bytes(literal_keep / "SKILL.md",
        "---\n"
        "name: literal-keep\n"
        "description: |+\n"
        "  Keep line one\n"
        "  Keep line two\n"
        "\n"
        "license: Apache-2.0\n"
        "---\n"
        "body");
    const fs::path folded = project / ".agents" / "skills" / "folded";
    fs::create_directories(folded);
    write_bytes(folded / "SKILL.md",
        "---\n"
        "name: folded\n"
        "description: >\n"
        "  Folded first\n"
        "  folded second\n"
        "\n"
        "  folded third\n"
        "license: Apache-2.0\n"
        "---\n"
        "body");
    const fs::path folded_strip = project / ".agents" / "skills" / "folded-strip";
    fs::create_directories(folded_strip);
    write_bytes(folded_strip / "SKILL.md",
        "---\n"
        "name: folded-strip\n"
        "description: >-\n"
        "  Folded strip first\n"
        "  folded strip second\n"
        "license: Apache-2.0\n"
        "---\n"
        "body");
    const fs::path folded_keep = project / ".agents" / "skills" / "folded-keep";
    fs::create_directories(folded_keep);
    write_bytes(folded_keep / "SKILL.md",
        "---\n"
        "name: folded-keep\n"
        "description: >+\n"
        "  Folded keep first\n"
        "  folded keep second\n"
        "\n"
        "license: Apache-2.0\n"
        "---\n"
        "body");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr catalog = do_get(skills);
    CHECK(catalog != nullptr);
    CHECK(catalog->status == 200);
    const json catalog_body = parse_body(catalog);
    std::map<std::string, std::string> descriptions;
    for (const auto & skill : catalog_body.at("skills")) {
        descriptions[skill.at("name").get<std::string>()] = skill.at("description").get<std::string>();
    }
    // literal keeps interior line breaks (including blank lines); clip keeps
    // one trailing break
    CHECK(descriptions.at("literal") == "First line\nSecond line\n\nFourth line\n");
    CHECK(descriptions.at("literal-strip") == "Strip line one\nStrip line two");
    CHECK(descriptions.at("literal-keep") == "Keep line one\nKeep line two\n\n");
    // folded turns single breaks into spaces and blank lines into paragraph breaks
    CHECK(descriptions.at("folded") == "Folded first folded second\nfolded third\n");
    CHECK(descriptions.at("folded-strip") == "Folded strip first folded strip second");
    CHECK(descriptions.at("folded-keep") == "Folded keep first folded keep second\n\n");

    // the read path surfaces the same parsed literal description in metadata
    const server_http_res_ptr read = do_post(skills, R"({"name":"literal"})");
    CHECK(read != nullptr);
    CHECK(read->status == 200);
    const json read_body = parse_body(read);
    CHECK(read_body.at("skill").at("metadata").at("description") == "First line\nSecond line\n\nFourth line\n");
}

static void test_provider_agents_not_rescanned() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);
    write_skill(home, "agents", "alpha", "global alpha");
    write_skill(project, "agents", "alpha", "project alpha");
    write_skill(project, "claude", "beta");

    // the configured provider list includes "agents": the built-in .agents root
    // must not be rescanned as a provider afterwards
    server_skills skills = make_skills(home, project, /* trust_project_skills */ true, {"agents", "claude"});

    const server_http_res_ptr catalog = do_get(skills);
    CHECK(catalog != nullptr);
    CHECK(catalog->status == 200);
    const json body = parse_body(catalog);
    CHECK(body.at("skills").size() == 2);
    CHECK(body.at("skills")[0].at("name") == "alpha");
    CHECK(body.at("skills")[0].at("scope") == "project");
    CHECK(body.at("skills")[0].at("provider") == "agents");
    CHECK(body.at("skills")[1].at("name") == "beta");
    CHECK(body.at("skills")[1].at("provider") == "claude");
    int shadowed = 0;
    for (const auto & item : body.at("diagnostics")) {
        if (item.at("code") == "skill_shadowed") {
            ++shadowed;
            // the only legitimate shadow is global alpha below project alpha;
            // a rescan of .agents as a provider would add a project-scoped one
            CHECK(item.at("scope") == "global");
        }
    }
    CHECK(shadowed == 1);

    const server_http_res_ptr read = do_post(skills, R"({"name":"alpha"})");
    CHECK(read != nullptr);
    CHECK(read->status == 200);
    const json read_body = parse_body(read);
    int read_shadowed = 0;
    for (const auto & item : read_body.at("diagnostics")) {
        if (item.at("code") == "skill_shadowed") {
            ++read_shadowed;
            CHECK(item.at("scope") == "global");
        }
    }
    CHECK(read_shadowed == 1);
}

static void test_suggestion_ranking_top_three_bounded() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path skill = write_skill(project, "agents", "ranked");
    for (const char * name : {"aaa.txt", "aab.txt", "bbb.txt", "ccc.txt"}) {
        write_bytes(skill / name, name);
    }

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    const server_http_res_ptr missing = do_post(skills, R"({"name":"ranked","path":"aax.txt"})");
    CHECK(missing != nullptr);
    CHECK(missing->status == 404);
    const json body = parse_body(missing);
    const json suggestions = body.at("suggestions");
    // at most three, best normalized Damerau-Levenshtein first, lexical tie-break
    CHECK(suggestions.size() == 3);
    CHECK(suggestions[0] == "aaa.txt");
    CHECK(suggestions[1] == "aab.txt");
    CHECK(suggestions[2] == "bbb.txt");
}

static void test_resource_path_length_bounded() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);

    const fs::path skill = write_skill(project, "agents", "bounded-path");
    const fs::path references = skill / "references";
    fs::create_directories(references);
    write_bytes(references / "DETAILS.md", "details");

    server_skills skills = make_skills(home, project, /* trust_project_skills */ true);

    // mirrors RESOURCE_PATH_MAX_BYTES in server-skills.cpp
    const size_t path_cap = 4096;

    // a missing path exactly at the cap is still ranked with suggestions
    const std::string at_cap = "references/DETAILS.md" + std::string(path_cap - std::string("references/DETAILS.md").size(), 'a');
    const server_http_res_ptr at_cap_missing = do_post(skills, "{\"name\":\"bounded-path\",\"path\":\"" + at_cap + "\"}");
    CHECK(at_cap_missing != nullptr);
    CHECK(at_cap_missing->status == 404);
    const json at_cap_body = parse_body(at_cap_missing);
    CHECK(at_cap_body.at("suggestions").size() == 1);
    CHECK(at_cap_body.at("suggestions")[0] == "references/DETAILS.md");

    // one byte over the cap is rejected before any ranking allocation
    const std::string over_cap = at_cap + "a";
    const server_http_res_ptr too_long = do_post(skills, "{\"name\":\"bounded-path\",\"path\":\"" + over_cap + "\"}");
    CHECK(too_long != nullptr);
    CHECK(too_long->status == 400);
}

// Fills the 32-entry catalog cache past its cap with 33 distinct effective
// CWDs, then revisits eviction candidates: every catalog must stay correct
// (its own fixture's skill), a recently revisited catalog must remain correct
// after an eviction candidate is revisited, and the counting token callback
// observes the LRU behavior through public results: a fresh CWD costs one
// generation probe plus one body measurement, a cache hit costs only the
// probe.
static void test_catalog_cache_lru_eviction() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    fs::create_directories(home);
    std::vector<fs::path> projects;
    for (int index = 0; index < 33; ++index) {
        const fs::path project = tmp / ("proj-" + std::to_string(index));
        fs::create_directories(project);
        write_skill(project, "agents", "skill-" + std::to_string(index),
                    "body-" + std::to_string(index), "desc-" + std::to_string(index));
        projects.push_back(project);
    }

    size_t measure_calls = 0;
    auto count_tokens = [&measure_calls](const std::string & text) -> std::optional<server_token_count_snapshot> {
        ++measure_calls;
        return server_token_count_snapshot{(text.size() + 3) / 4, 0};
    };
    server_skills skills = make_skills(home, projects[0], true, {"agents"}, token_count_callback(count_tokens));

    auto check_skill = [&](int index) {
        const server_http_res_ptr response = do_get(skills, {{"X-Skill-Cwd", projects[index].string()}});
        CHECK(response != nullptr);
        CHECK(response->status == 200);
        const json body = parse_body(response);
        CHECK(body.at("skills").size() == 1);
        const json skill = body.at("skills")[0];
        CHECK(skill.at("name") == "skill-" + std::to_string(index));
        CHECK(skill.at("description") == "desc-" + std::to_string(index));
        const std::string body_text = "body-" + std::to_string(index);
        const size_t bytes = body_text.size();
        CHECK(skill.at("instruction").at("bytes") == bytes);
        CHECK(skill.at("instruction").at("tokens") == (bytes + 3) / 4);
        CHECK(skill.at("instruction").at("tokens_estimated") == false);
    };

    // fill the cache past the 32-entry cap: the 33rd fill evicts proj-0
    for (int index = 0; index < 33; ++index) {
        check_skill(index);
    }
    CHECK(measure_calls == 66); // 33 requests x (generation probe + body)

    // revisit the eviction candidate: refilled (probe + body), then a hit
    check_skill(0);
    CHECK(measure_calls == 68);
    check_skill(0);
    CHECK(measure_calls == 69);
    check_skill(1); // an eviction candidate by now; its refill evicts proj-2
    CHECK(measure_calls == 71);
    // recently revisited catalogs remain correct (hits, probe only)
    check_skill(32);
    check_skill(4);
    CHECK(measure_calls == 73);

    // every CWD still resolves to its own catalog. The cache holds all 33
    // except proj-2, but the sweep refills it in ascending order, so the
    // eviction cascades: proj-2's refill evicts proj-3, which then misses on
    // the next iteration, and so on through proj-32. The 33 requests hit
    // proj-0, proj-1, proj-4 (probe only) and miss the other 30 (probe +
    // measurement), a deterministic sequence regardless of thread timing.
    for (int index = 0; index < 33; ++index) {
        check_skill(index);
    }
    CHECK(measure_calls == 136); // 73 + 3 hits + 30 misses x 2 calls
}

// Mutates SKILL.md in place and asserts the catalog refreshes bytes/body
// values without an identity change; the counting callback proves the
// unchanged-file request is served from the cache (probe only) while the
// mutation triggers a re-read and re-measurement.
static void test_catalog_cache_mutation_freshness() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);
    const fs::path skill = write_skill(project, "agents", "mutable", "first body", "first desc");

    size_t measure_calls = 0;
    auto count_tokens = [&measure_calls](const std::string & text) -> std::optional<server_token_count_snapshot> {
        ++measure_calls;
        return server_token_count_snapshot{(text.size() + 3) / 4, 7};
    };
    server_skills skills = make_skills(home, project, true, {"agents"}, token_count_callback(count_tokens));

    const server_http_res_ptr first = do_get(skills);
    CHECK(first != nullptr && first->status == 200);
    const json first_skill = parse_body(first).at("skills")[0];
    const std::string first_id = first_skill.at("id").get<std::string>();
    CHECK(first_skill.at("name") == "mutable");
    CHECK(first_skill.at("description") == "first desc");
    CHECK(first_skill.at("instruction").at("bytes") == std::string("first body").size());
    CHECK(first_skill.at("instruction").at("tokens") == (std::string("first body").size() + 3) / 4);
    CHECK(first_skill.at("instruction").at("tokens_estimated") == false);
    CHECK(measure_calls == 2); // probe + body

    // unchanged file: cache hit, only the generation probe runs
    const server_http_res_ptr second = do_get(skills);
    CHECK(second != nullptr && second->status == 200);
    const json second_skill = parse_body(second).at("skills")[0];
    CHECK(second_skill.at("id") == first_id);
    CHECK(second_skill.at("instruction").at("bytes") == std::string("first body").size());
    CHECK(measure_calls == 3); // probe only

    // in-place mutation: observable file state changes, values refresh
    // without an identity change
    write_bytes(skill / "SKILL.md", "---\nname: mutable\ndescription: second desc\n---\nsecond body");
    const server_http_res_ptr third = do_get(skills);
    CHECK(third != nullptr && third->status == 200);
    const json third_skill = parse_body(third).at("skills")[0];
    CHECK(third_skill.at("id") == first_id);
    CHECK(third_skill.at("name") == "mutable");
    CHECK(third_skill.at("description") == "second desc");
    CHECK(third_skill.at("instruction").at("bytes") == std::string("second body").size());
    CHECK(third_skill.at("instruction").at("tokens") == (std::string("second body").size() + 3) / 4);
    CHECK(third_skill.at("instruction").at("tokens_estimated") == false);
    CHECK(measure_calls == 5); // probe + re-measured body
}

// The cache key includes the tokenizer generation: replacing the tokenizer
// (model reload during the load/sleep lifecycle) must invalidate entries
// measured against the old generation instead of serving stale counts.
static void test_catalog_cache_generation_invalidation() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    const fs::path project = tmp / "project";
    fs::create_directories(home);
    fs::create_directories(project);
    write_skill(project, "agents", "gen-skill", "body", "desc");

    uint64_t generation = 1;
    auto count_tokens = [&generation](const std::string & text) -> std::optional<server_token_count_snapshot> {
        return server_token_count_snapshot{(text.size() + 3) / 4 + (generation - 1), generation};
    };
    server_skills skills = make_skills(home, project, true, {"agents"}, token_count_callback(count_tokens));

    const server_http_res_ptr first = do_get(skills);
    CHECK(first != nullptr && first->status == 200);
    const json first_skill = parse_body(first).at("skills")[0];
    const size_t first_tokens = first_skill.at("instruction").at("tokens").get<size_t>();
    CHECK(first_tokens == (std::string("body").size() + 3) / 4);
    CHECK(first_skill.at("instruction").at("tokens_estimated") == false);

    // tokenizer/model replaced: the generation advances, so the entry cached
    // under the old generation must not be served
    generation = 2;
    const server_http_res_ptr second = do_get(skills);
    CHECK(second != nullptr && second->status == 200);
    const json second_skill = parse_body(second).at("skills")[0];
    CHECK(second_skill.at("instruction").at("tokens") == first_tokens + 1);
    CHECK(second_skill.at("instruction").at("tokens_estimated") == false);

    // the new-generation entry is now cached and served
    const server_http_res_ptr third = do_get(skills);
    CHECK(third != nullptr && third->status == 200);
    CHECK(parse_body(third).at("skills")[0].at("instruction").at("tokens") == first_tokens + 1);
}

// The catalog cache is shared by concurrent HTTP worker threads: drive the
// public GET handler from several threads at once, mixing hits, misses, and
// eviction-triggering fills, and assert every response resolves to its own
// fixture. The cache must not crash, deadlock, or return cross-contaminated
// data under this churn (eviction while another handler is still using an
// entry would otherwise be a use-after-free).
static void test_catalog_cache_concurrent_gets() {
    const fs::path tmp = make_temp_dir();
    const fs::path home = tmp / "home";
    fs::create_directories(home);
    std::vector<fs::path> projects;
    for (int index = 0; index < 33; ++index) {
        const fs::path project = tmp / ("conc-" + std::to_string(index));
        fs::create_directories(project);
        write_skill(project, "agents", "skill-" + std::to_string(index),
                    "body-" + std::to_string(index), "desc-" + std::to_string(index));
        projects.push_back(project);
    }

    server_skills skills = make_skills(home, projects[0], true, {"agents"});

    std::atomic<size_t> failures = 0;
    auto worker = [&](int seed) {
        for (int round = 0; round < 5; ++round) {
            for (int index = 0; index < 33; ++index) {
                const int effective = (seed + index) % 33;
                const server_http_res_ptr response = do_get(skills, {{"X-Skill-Cwd", projects[effective].string()}});
                if (response == nullptr || response->status != 200) {
                    ++failures;
                    continue;
                }
                const json body = parse_body(response);
                if (body.at("skills").size() != 1 ||
                    body.at("skills")[0].at("name") != "skill-" + std::to_string(effective)) {
                    ++failures;
                }
            }
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back(worker, t * 7);
    }
    for (auto & thread : threads) {
        thread.join();
    }
    CHECK(failures == 0);
}

int main() {
    try {
        const fs::path original_cwd = fs::current_path();

        test_catalog_global_only();
        test_catalog_precedence_and_cwd_identity();
        test_catalog_rejects_invalid_cwd_and_escapes_xml();
        test_catalog_diagnoses_invalid_candidates();
        test_reads_current_base_and_resources();
        test_read_rejects_unsafe_paths_and_ranks_suggestions();
        test_resource_listing_bounds_and_invalid_content();
        test_rejects_directory_and_file_symlink_escapes();
        test_configured_root_symlink_containment();
        test_cosmetic_name_diagnostics();
        test_cosmetic_name_leading_trailing_hyphen();
        test_block_scalar_descriptions();
        test_provider_agents_not_rescanned();
        test_suggestion_ranking_top_three_bounded();
        test_resource_path_length_bounded();
        test_catalog_cache_lru_eviction();
        test_catalog_cache_mutation_freshness();
        test_catalog_cache_generation_invalidation();
        test_catalog_cache_concurrent_gets();

        std::error_code ec;
        fs::current_path(original_cwd, ec);
        for (const fs::path & root : g_temp_roots) {
            fs::remove_all(root, ec);
        }
    } catch (const std::exception & e) {
        std::cerr << "test-server-skills: unexpected exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test-server-skills: all checks passed\n";
    return 0;
}
