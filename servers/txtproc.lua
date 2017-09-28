local ffi = require("ffi")
local E4SMT = ffi.load("./servers/lib/libTargomanTextProcessor.so")

ffi.cdef("bool c_init(const char* _normalizationFilePath,const char* _abbreviationFilePath,const char* _spellCorrectorBaseConfigPath,const char* _language);")
ffi.cdef("bool c_ixml2Text(const char* _source, char* _target, int _targetMaxLength);")
ffi.cdef("bool c_text2IXML(const char* _source, const char* _language, bool _noSpellCorrector, char* _target, int _targetMaxLength);")
ffi.cdef("bool c_tokenize(const char* _source, const char* _language, bool _noSpellCorrector, char* _target, int _targetMaxLength);")
ffi.cdef("bool c_normalize(const char* _source, const char* _language, char* _target, int _targetMaxLength);")

local function init(_normalizationFilePath, _abbreviationFilePath, _spellCorrectorBaseConfigPath, _language)
    return E4SMT.c_init(_normalizationFilePath, _abbreviationFilePath, _spellCorrectorBaseConfigPath, _language)
end

local function ixml2Text(_source)
    local BuffSize = 8 * #_source
    local Buffer = ffi.new('char[?]', BuffSize)
    E4SMT.c_ixml2Text(_source, Buffer, BuffSize)
    return ffi.string(Buffer)
end

local function text2IXML(_source, _language, _noSpellCorrector)
    local BuffSize = 8 * #_source
    local Buffer = ffi.new('char[?]', BuffSize)
    E4SMT.c_text2IXML(_source, _language, _noSpellCorrector, Buffer, BuffSize)
    return ffi.string(Buffer)
end

local function tokenize(_source, _language, _noSpellCorrector)
    local BuffSize = 8 * #_source
    local Buffer = ffi.new('char[?]', BuffSize)
    E4SMT.c_tokenize(_source, _language, _noSpellCorrector, Buffer, BuffSize)
    return ffi.string(Buffer)
end

local function normalize(_source, _language, _noSpellCorrector)
    local BuffSize = 8 * #_source
    local Buffer = ffi.new('char[?]', BuffSize)
    E4SMT.c_normalize(_source, _language, Buffer, BuffSize)
    return ffi.string(Buffer)
end

init('./Normalization.conf', './Abbreviations.tbl', './', '')

txtproc = {
    ixml2Text = ixml2Text,
    text2IXML = text2IXML,
    tokenize = tokenize,
    normalize = normalize
}