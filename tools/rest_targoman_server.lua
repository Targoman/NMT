#!/usr/bin/env lua
--[[
This requires the restserver-xavante rock to run.
run server (this file)
th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1
query the server:
curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "international migration" }]' http://127.0.0.1:7784/translator/translate
]]

require('onmt.init')

local restserver = require("restserver")

local cmd = onmt.utils.ExtendedCmdLine.new('rest_targoman_server.lua')

local options = {
    {'-port', '8000', [[Port to run the server on.]]},
}

cmd:setCmdLineOptions(options, 'Server')

onmt.translate.Translator.declareOpts(cmd)

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text("")
cmd:text("Other options")
cmd:text("")

cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory.]])

local ffi = require("ffi")
local E4SMT = ffi.load("./Targoman/E4SMT4Lua.so")

ffi.cdef("void tokenize(const char*, char*, int);")

local opt = cmd:parse(arg)
local translator

local function tokenize(lines)
    local batch = {}
    local normals = {}

    for i = 1, #lines do

        local buffSize = 9 * #lines[i].src
        local buffer = ffi.new('char[?]', buffSize)
        E4SMT.tokenize(lines[i].src, buffer, buffSize)
        local normalLine = ffi.string(buffer)

        for s in normalLine:gmatch("[^\r\n]+") do
            table.insert(normals, s)
        end

    end

    for i = 1, #normals do

        local tokens = {}
        for word in normals[i]:gmatch'([^%s]+)' do
            table.insert(tokens, word)
        end
        table.insert(batch, translator:buildInput(tokens))
    end

    collectgarbage()

    return batch
end

local function buildResultObject(batch, rawResults)

    local results = {
        base = {},
        phrases = {},
        alignments = {}
    }

    for i = 1, #batch do
        table.insert(results.base, {
            translator:buildOutput(rawResults[i].preds[1]),
            translator:buildOutput(batch[i])
        })


        local function getWordMapping(attention)
            attention = torch.cat(attention, 2)
            for j = 1, #batch[i].words do
                local r = attention:narrow(1, j, 1)
                r:copy(r / r:max())
            end
            local _, wordmapping = torch.max(attention, 1)
            local wordmapping = torch.totable(wordmapping:storage())
            return wordmapping
        end

        local wordmapping = getWordMapping(rawResults[i].preds[1].attention)
        local words = onmt.utils.Features.annotate(rawResults[i].preds[1].words, rawResults[i].preds[1].features)

        local phrases = {}
        local alignments = {}

        local phraseIndex = 0
        local lastIndex
        for j, index in ipairs(wordmapping) do
            if index == lastIndex then
                phrases[#phrases][1] = phrases[#phrases][1] .. ' ' .. words[j]
                alignments[#alignments][3][1][1] = alignments[#alignments][3][1][1] .. ' ' .. words[j]
            else
                table.insert(phrases, {
                    words[j],
                    phraseIndex
                })
                table.insert(alignments, {
                    batch[i].words[index],
                    index,
                    {
                        { words[j], true}
                    }
                })
                phraseIndex = phraseIndex + 1
            end
            lastIndex = index
        end
        
        for j = 2, #rawResults[i].preds do
            wordmapping = getWordMapping(rawResults[i].preds[j].attention)
            words = onmt.utils.Features.annotate(rawResults[i].preds[j].words, rawResults[i].preds[j].features)
            for k, index in ipairs(wordmapping) do
                for l = 1, #alignments do
                    if alignments[l][2] == index then
                        local found = false
                        for m = 1, #alignments[l][3] do
                            if alignments[l][3][m][1] == words[k] then
                                found = true
                                break
                            end
                        end
                        if not found then
                            table.insert(alignments[l][3], { words[k], false})
                        end
                    end
                end
            end
        end

        table.insert(results.phrases, phrases)
        table.insert(results.alignments, alignments)
    end

    return { t = { results.base, results.phrases, results.alignments } }
end

local function translateMessage(translator, lines)

    local batch

    -- We need to tokenize the input line before translation
    _G.logger:debug("Start Tokenization")
    batch = tokenize(lines)

    -- Translate
    _G.logger:debug("Start Translation")
    local results = translator:translate(batch)
    _G.logger:debug("End Translation")

    -- Return the nbest translations for each in the batch.
    return buildResultObject(batch, results)
end

local function init_server(port, translator)
    local server = restserver:new():port(port)

    server:add_resource("translator", {
        {
        method = "POST",
        path = "/translate",
        consumes = "application/json",
        produces = "application/json",
        handler = function(req)
            _G.logger:debug("receiving request")
            local translate = translateMessage(translator, req)
            _G.logger:debug("sending response")
            return restserver.response():status(200):entity(translate)
        end,
        }
    })
    return server
end

local function main()
    -- load logger
    _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
    onmt.utils.Cuda.init(opt)

    -- disable profiling
    _G.profiler = onmt.utils.Profiler.new(false)

    _G.logger:info("Loading model")
    translator = onmt.translate.Translator.new(opt)
    _G.logger:info("Launch server")
    local server = init_server(opt.port, translator)
    -- This loads the restserver.xavante plugin
    server:enable("restserver.xavante"):start()
end

main()
