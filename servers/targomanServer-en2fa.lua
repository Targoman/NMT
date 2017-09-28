#!/usr/bin/env luajit

require('onmt.init')
require("socket")
require('servers.txtproc')

print('Updated by Mehran and Behrooz')
local cmd = onmt.utils.ExtendedCmdLine.new('turbo_targoman_server.lua')

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

cmd:option('-BatchRequestssize', 1000, [[Size of each parallel BatchRequests - you should not change except if low memory.]])

local opt = cmd:parse(arg)
local translator

local function myIPAddress()
    local s = socket.udp()
    s:setpeername("74.125.115.104",80)
    local ip, _ = s:getsockname()
    return ip
end

local function allUpperCase(seq)
    for i=1, #seq do
        c = string.sub(seq,i,i)
        if string.match(c,"%l") then
            return false
        end
    end
    return true
end

local function replaceUnkWord(labelToIdx, token)
    if labelToIdx[token] == nil and not allUpperCase(token) then
        seq = token:lower()
        if labelToIdx[seq] ~= nil then
            return seq
        end
    end

    return token
end


local function tokenize(lines)
    local BatchRequests = {}
    local Paragraphs = {}
    local j = 0

    for i = 1, #lines do
        local NormalizedLine = txtproc.tokenize(lines[i], 'en', false)
        for Sentence in NormalizedLine:gmatch("[^\r\n]+") do
            j=j+1
            local tokens = {}
            for word in Sentence:gmatch'([^%s]+)' do
                table.insert(tokens, replaceUnkWord(translator.dicts.src.words.labelToIdx,word))
            end
            table.insert(BatchRequests, translator:buildInput(tokens))
        end
        table.insert(Paragraphs, j)
    end

    collectgarbage()

    return {BatchRequests,Paragraphs}
end

local function postProcess(iString)
    iString = iString:gsub('1', '۱');
    iString = iString:gsub('2', '۲');
    iString = iString:gsub('3', '۳');
    iString = iString:gsub('4', '۴');
    iString = iString:gsub('5', '۵');
    iString = iString:gsub('6', '۶');
    iString = iString:gsub('7', '۷');
    iString = iString:gsub('8', '۸');
    iString = iString:gsub('9', '۹');
    iString = iString:gsub('0', '۰');

    iString = iString:gsub('  ', ' ');
    iString = iString:gsub('  ', ' ');
    iString = iString:gsub(' %.', '.');
    iString = iString:gsub(' ,', '،');
    iString = iString:gsub(' ;', '؛');
    iString = iString:gsub(' %?', '؟');

    return iString;
end

local function buildResultObject(BatchRequests, rawResults, paragraphs)

    local results = {
        base = {},
        phrases = {},
        alignments = {}
    }

    for i=1, #paragraphs do
        table.insert(results.base, {})
        table.insert(results.phrases, {})
        table.insert(results.alignments, {})
    end

    local ParIndex = 1

    for i = 1, #BatchRequests do
        if(i > paragraphs[ParIndex]) then
            ParIndex = ParIndex + 1;
        end

        local function getWordMapping(attention)
            attention = torch.cat(attention, 2)
            for j = 1, #BatchRequests[i].words do
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
                if(true or words[j] ~= BatchRequests[i].words[index] or string.match(words[j], "%p") ~= nil) then
                    table.insert(alignments, {
                        BatchRequests[i].words[index],
                        index,
                        {
                            { postProcess(words[j]), true}
                        }
                    })
                else
                    table.insert(alignments, {
                        BatchRequests[i].words[index],
                        index,
                        {
                            { '', true}
                        }
                    })
                end
                phraseIndex = phraseIndex + 1
            end
            lastIndex = index
        end

        for j = 2, #rawResults[i].preds do
            wordmapping = getWordMapping(rawResults[i].preds[j].attention)
            words = onmt.utils.Features.annotate(rawResults[i].preds[j].words, rawResults[i].preds[j].features)
            local dummy = {}
            lastIndex = -1
            for k, index in ipairs(wordmapping) do
                if not dummy[index] then
                    dummy[index] = { words[k] }
                else
                    if lastIndex == index then
                        dummy[index][#dummy[index]] = dummy[index][#dummy[index]] .. ' ' .. words[k]
                    else
                        table.insert(dummy[index], words[k])
                    end
                end
                lastIndex = index
            end

            for index, ks in pairs(dummy) do
                for l = 1, #alignments do
                    if alignments[l][2] == index then
                        for phraseIndex, phrase in pairs(ks) do
                            local found = false
                            for m = 1, #alignments[l][3] do
                                if alignments[l][3][m][1] == '' and phrase ~= alignments[l][1] then
                                    alignments[l][3][m][1] = postProcess(phrase)
                                    phrases[l][1]=phrase
                                    found = true
                                    break;
                                elseif alignments[l][3][m][1] == phrase or phrase == alignments[l][1] then
                                    found = true
                                    break
                                end
                            end
                            if not found then
                                table.insert(alignments[l][3], { postProcess(phrase), false})
                            end
                        end
                    end
                end
            end
        end

        for l = 1, #alignments do
            if alignments[l][3][1][1] == '' then
                alignments[l][3][1][1] = postProcess(alignments[l][1])
                phrases[l][1] = alignments[l][1]
            end
        end

        local Alignments = alignments

        local Default = {};
        table.insert(Default, 0);
        table.insert(Default, 0);
        table.insert(Default, {0,0});
        for NGram=1,5 do
            local ToCheck = {};
            local ProcessedAlignments = {};
            for i=1,NGram do
                table.insert(ToCheck, Default);
            end

            local i=0
            while i < #Alignments do
                i=i+1
                local Similar = true
                if(NGram + i - 1 < #Alignments) then
                    for j=1,NGram do
                        if ToCheck[j][1] ~= Alignments[j + i - 1][1] or ToCheck[j][3][1][1] ~= Alignments[j + i - 1][3][1][1] then
                            Similar = false;
                            break;
                        end
                    end
                else
                    Similar = false
                end
                if Similar == false then
                    table.insert(ProcessedAlignments, Alignments[i])
                    for j=1,NGram - 1 do
                        ToCheck[j] = ToCheck[j+1];
                    end
                    ToCheck[NGram] = Alignments[i];
                else
                    i = i + NGram -1;
                end
            end
            Alignments = ProcessedAlignments;
        end

        phrases = {}
        FinalString = ''
        for j=1, #Alignments do
            FinalString = FinalString .. Alignments[j][3][1][1] .. ' '
            table.insert(phrases, {postProcess(Alignments[j][3][1][1]), j-1})
        end

        if #results.base[ParIndex] > 0 then
            results.base[ParIndex][1] =  postProcess(results.base[ParIndex][1] .. ' ' .. FinalString);
            results.base[ParIndex][2] =  results.base[ParIndex][2] .. ' ' .. translator:buildOutput(BatchRequests[i]);

            local lastCount = #results.phrases[ParIndex]
            for j=1, #phrases do
                phrases[j][2] = phrases[j][2] + lastCount;
                table.insert(results.phrases[ParIndex],phrases[j])

            end
            local lastCount = #results.alignments[ParIndex]
            for j=1, #Alignments do
                Alignments[j][2] = Alignments[j][2] + lastCount;
                table.insert(results.alignments[ParIndex],Alignments[j])
            end
        else
            results.base[ParIndex] =  {
                postProcess(FinalString),
                translator:buildOutput(BatchRequests[i])
            }
            results.phrases[ParIndex] =  phrases
            results.alignments[ParIndex] = Alignments
        end
    end

    return { serverIP = myIPAddress(),t = { results.base, results.phrases, results.alignments } }
end

local function split2lines(str)
    local t = {}
    local function helper(line)
        table.insert(t, line)
        return ""
    end
    helper((str:gsub("(.-)\r?\n", helper)))
    return t
end

local function translateMessage(translator, lines)

    local BatchRequests
    local Paragraphs

    -- We need to tokenize the input line before translation
    _G.logger:debug("Start Tokenization")
    if #lines == 1 then
        local Source = lines[1].src
        _G.logger:debug(Source)
        lines = split2lines(Source)
    end

    Result = tokenize(lines)
    BatchRequests = Result[1]
    Paragraphs = Result[2]


    -- Translate
    _G.logger:debug("Start Translation")
    local Results = translator:translate(BatchRequests)
    _G.logger:debug("End Translation")

    -- Return the nbest translations for each in the BatchRequests.
    return buildResultObject(BatchRequests, Results, Paragraphs)
end

local turbo = require "turbo"

local ExampleHandler = class("ExampleHandler", turbo.web.RequestHandler)
local cjson = require "cjson"

function ExampleHandler:post()
    print("Request received =======================================")
    _G.logger:debug("receiving request")
    local translate = translateMessage(translator, cjson.decode(self.request.body))
    _G.logger:debug("sending response")
    print("======================================= Request Processed")
    self:write(translate)
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

    turbo.web.Application({{"^/translator/translate$", ExampleHandler}}):listen(tonumber(opt.port))
    turbo.ioloop.instance():start()
end

main()
